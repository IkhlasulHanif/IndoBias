"""Command-line interface for the IndoBias task complexity framework."""

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from .config import DEFAULT_OUTPUT_DIR, DIMENSIONS_FILE, GROUPS_DIR, VALID_DEMOGRAPHIES, VALID_DIMENSIONS
from .generator import (
    PromptExporter,
    count_prompts,
    generate_prompts,
    load_dimension_catalog,
    load_groups_from_csv,
    load_groups_from_directory,
)
from .data_models import Demography, StereotypeDimension
from .tasks import ALL_TASKS, get_task

app = typer.Typer(
    name="indobias",
    help="Task complexity framework for testing stereotype expression in LLMs",
    add_completion=False,
)
console = Console()


def format_valid_options(values: tuple[str, ...]) -> str:
    """Format a list of valid CLI options for error/help messages."""
    return ", ".join(values)


@app.command()
def generate(
    dimensions_file: Annotated[
        Path,
        typer.Argument(help="Path to dimensions catalog Python file"),
    ] = DIMENSIONS_FILE,
    groups_path: Annotated[
        Path,
        typer.Option("--groups", "-g", help="Path to groups CSV file or directory"),
    ] = GROUPS_DIR,
    output_dir: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory for generated prompts"),
    ] = DEFAULT_OUTPUT_DIR,
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: files, jsonl, json"),
    ] = "jsonl",
    tasks: Annotated[
        Optional[str],
        typer.Option("--tasks", "-t", help="Comma-separated task IDs (e.g., '1,3,5') or 'all'"),
    ] = None,
    demography: Annotated[
        Optional[str],
        typer.Option("--demography", "-d", help="Filter by demography"),
    ] = None,
    dimension: Annotated[
        Optional[str],
        typer.Option("--dimension", "-D", help="Filter by dimension"),
    ] = None,
    no_metadata: Annotated[
        bool,
        typer.Option("--no-metadata", help="Exclude metadata from individual files"),
    ] = False,
) -> None:
    """Generate prompts from dimensions and groups."""
    
    # Load dimension catalog
    console.print(f"[cyan]Loading dimensions from {dimensions_file}...[/cyan]")
    
    if not dimensions_file.exists():
        console.print(f"[red]Error: Dimensions file not found: {dimensions_file}[/red]")
        raise typer.Exit(1)
    
    catalog = load_dimension_catalog(dimensions_file)
    console.print(f"[green]✓[/green] Loaded {len(catalog.dimensions)} dimensions")
    
    # Load groups
    console.print(f"[cyan]Loading groups from {groups_path}...[/cyan]")
    
    if not groups_path.exists():
        console.print(f"[red]Error: Groups path not found: {groups_path}[/red]")
        raise typer.Exit(1)
    
    if groups_path.is_dir():
        groups = load_groups_from_directory(groups_path)
    else:
        groups = load_groups_from_csv(groups_path)
    
    console.print(f"[green]✓[/green] Loaded {len(groups.groups)} groups")
    
    # Parse task IDs
    task_ids = None
    if tasks and tasks.lower() != "all":
        try:
            task_ids = [int(t.strip()) for t in tasks.split(",")]
            for tid in task_ids:
                if tid < 1 or tid > 7:
                    raise ValueError(f"Invalid task ID: {tid}")
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)
    
    # Parse demography filter
    demographies = None
    if demography:
        try:
            demographies = [Demography(demography.lower())]
        except ValueError:
            console.print(f"[red]Error: Invalid demography '{demography}'[/red]")
            console.print(f"Valid options: {format_valid_options(VALID_DEMOGRAPHIES)}")
            raise typer.Exit(1)
    
    # Parse dimension filter
    dimensions_filter = None
    if dimension:
        try:
            dimensions_filter = [StereotypeDimension(dimension.lower())]
        except ValueError:
            console.print(f"[red]Error: Invalid dimension '{dimension}'[/red]")
            console.print(f"Valid options: {format_valid_options(VALID_DIMENSIONS)}")
            raise typer.Exit(1)
    
    # Count prompts
    total = count_prompts(groups, catalog, task_ids, demographies, dimensions_filter)
    console.print(f"  Total prompts to generate: {total}")
    
    if total == 0:
        console.print("[yellow]No prompts to generate with the specified filters.[/yellow]")
        raise typer.Exit(0)
    
    # Generate and export
    exporter = PromptExporter(output_dir)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Generating prompts...", total=None)
        
        prompts = generate_prompts(groups, catalog, task_ids, demographies, dimensions_filter)
        
        if format == "files":
            count = exporter.export_individual_files(prompts, include_metadata=not no_metadata)
        elif format == "jsonl":
            count = exporter.export_jsonl(prompts, organize_by_dimension=True)
        elif format == "json":
            count = exporter.export_json(prompts)
        else:
            console.print(f"[red]Error: Unknown format '{format}'[/red]")
            raise typer.Exit(1)
    
    console.print(f"[green]✓[/green] Generated {count} prompts to {output_dir}")


@app.command()
def list_tasks() -> None:
    """List all available task templates with their complexity levels."""
    
    table = Table(title="Task Complexity Levels", show_header=True)
    table.add_column("ID", style="cyan", justify="center")
    table.add_column("Name", style="green")
    table.add_column("Level", justify="center")
    table.add_column("Rationale", style="dim", max_width=50)
    
    for task in ALL_TASKS:
        rationale = task.rationale.replace("\n", " ").strip()[:50] + "..."
        table.add_row(
            str(task.task_id),
            task.name,
            str(task.complexity_level),
            rationale,
        )
    
    console.print(table)


@app.command()
def list_dimensions(
    dimensions_file: Annotated[
        Path,
        typer.Argument(help="Path to dimensions catalog Python file"),
    ] = DIMENSIONS_FILE,
) -> None:
    """List all stereotype dimensions and their pairs."""
    
    if not dimensions_file.exists():
        console.print(f"[red]Error: Dimensions file not found: {dimensions_file}[/red]")
        raise typer.Exit(1)
    
    catalog = load_dimension_catalog(dimensions_file)
    
    console.print(Panel(
        f"[bold]{catalog.name}[/bold] v{catalog.version}\n"
        f"[dim]{catalog.description or 'No description'}[/dim]",
        title="Dimension Catalog",
    ))
    
    for dim in catalog.dimensions:
        table = Table(
            title=f"{dim.name} ({dim.dimension.value.upper()})",
            show_header=True,
        )
        table.add_column("Pair ID", style="cyan")
        table.add_column("Positive", style="green")
        table.add_column("Negative", style="red")
        table.add_column("Description", style="dim", max_width=30)
        
        for pair in dim.pairs:
            table.add_row(
                pair.pair_id,
                pair.positive_label,
                pair.negative_label,
                pair.description or "",
            )
        
        console.print(table)
        console.print()


@app.command()
def list_groups(
    groups_path: Annotated[
        Path,
        typer.Argument(help="Path to groups CSV file or directory"),
    ] = GROUPS_DIR,
    demography: Annotated[
        Optional[str],
        typer.Option("--demography", "-d", help="Filter by demography"),
    ] = None,
) -> None:
    """List all demographic groups from CSV files."""
    
    if not groups_path.exists():
        console.print(f"[red]Error: Groups path not found: {groups_path}[/red]")
        raise typer.Exit(1)
    
    if groups_path.is_dir():
        groups = load_groups_from_directory(groups_path)
    else:
        groups = load_groups_from_csv(groups_path)
    
    # Filter by demography if specified
    filtered = groups.groups
    if demography:
        try:
            dem = Demography(demography.lower())
            filtered = groups.get_groups_by_demography(dem)
        except ValueError:
            console.print(f"[red]Error: Invalid demography '{demography}'[/red]")
            console.print(f"Valid options: {format_valid_options(VALID_DEMOGRAPHIES)}")
            raise typer.Exit(1)
    
    # Group by demography for display
    by_demography: dict[str, list] = {}
    for group in filtered:
        dem = group.demography.value
        if dem not in by_demography:
            by_demography[dem] = []
        by_demography[dem].append(group)
    
    for dem, dem_groups in sorted(by_demography.items()):
        table = Table(title=f"{dem.upper()} ({len(dem_groups)} groups)", show_header=True)
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Country", style="dim")
        
        for group in dem_groups:
            table.add_row(group.group_id, group.group_name, group.country or "")
        
        console.print(table)
        console.print()


@app.command()
def show_task(
    task_id: Annotated[int, typer.Argument(help="Task ID (1-7)")],
) -> None:
    """Show detailed information about a specific task."""
    
    try:
        task = get_task(task_id)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    
    console.print(Panel(
        f"[bold cyan]{task.name}[/bold cyan]\n\n"
        f"[yellow]Complexity Level:[/yellow] {task.complexity_level}/7\n\n"
        f"[yellow]Rationale:[/yellow]\n{task.rationale}\n\n"
        f"[yellow]Template:[/yellow]\n[dim]{task.template}[/dim]",
        title=f"Task {task.task_id}",
        expand=False,
    ))


@app.command()
def preview(
    dimensions_file: Annotated[
        Path,
        typer.Option("--dimensions", help="Path to dimensions catalog Python file"),
    ] = DIMENSIONS_FILE,
    groups_path: Annotated[
        Path,
        typer.Option("--groups", "-g", help="Path to groups CSV file or directory"),
    ] = GROUPS_DIR,
    task_id: Annotated[
        int,
        typer.Option("--task", "-t", help="Task ID to preview (1-7)"),
    ] = 1,
    group_id: Annotated[
        Optional[str],
        typer.Option("--group", help="Specific group ID to preview"),
    ] = None,
    dimension: Annotated[
        Optional[str],
        typer.Option("--dimension", "-D", help="Filter by dimension"),
    ] = None,
) -> None:
    """Preview generated prompts without saving to files."""
    
    # Load catalog
    if not dimensions_file.exists():
        console.print(f"[red]Error: Dimensions file not found: {dimensions_file}[/red]")
        raise typer.Exit(1)
    
    catalog = load_dimension_catalog(dimensions_file)
    
    # Load groups
    if not groups_path.exists():
        console.print(f"[red]Error: Groups path not found: {groups_path}[/red]")
        raise typer.Exit(1)
    
    if groups_path.is_dir():
        groups = load_groups_from_directory(groups_path)
    else:
        groups = load_groups_from_csv(groups_path)
    
    # Get task
    try:
        task = get_task(task_id)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    
    # Filter dimension
    dims_to_show = catalog.dimensions
    if dimension:
        try:
            dim_filter = StereotypeDimension(dimension.lower())
            dims_to_show = [d for d in dims_to_show if d.dimension == dim_filter]
        except ValueError:
            console.print(f"[red]Error: Invalid dimension '{dimension}'[/red]")
            console.print(f"Valid options: {format_valid_options(VALID_DIMENSIONS)}")
            raise typer.Exit(1)
    
    # Get groups to preview
    if group_id:
        group = groups.get_group(group_id)
        if not group:
            console.print(f"[red]Error: Group '{group_id}' not found[/red]")
            raise typer.Exit(1)
        preview_groups = [group]
    else:
        preview_groups = groups.groups[:3]  # First 3 groups
    
    console.print(f"[cyan]Previewing Task {task_id}: {task.name}[/cyan]\n")
    
    for group in preview_groups:
        for dim in dims_to_show[:1]:  # Show first dimension only
            for pair in dim.pairs[:2]:  # First 2 pairs
                prompt = task.format_prompt(
                    group=group.group_name,
                    pos=pair.positive_label,
                    neg=pair.negative_label,
                    country=group.country,
                )
                
                console.print(Panel(
                    prompt,
                    title=f"[green]{group.group_name}[/green] | [magenta]{dim.dimension.value.upper()}[/magenta] | {pair.pair_id}",
                    border_style="dim",
                ))
                console.print()


@app.command()
def stats(
    dimensions_file: Annotated[
        Path,
        typer.Option("--dimensions", help="Path to dimensions catalog Python file"),
    ] = DIMENSIONS_FILE,
    groups_path: Annotated[
        Path,
        typer.Option("--groups", "-g", help="Path to groups CSV file or directory"),
    ] = GROUPS_DIR,
) -> None:
    """Show statistics about dimensions and groups."""
    
    # Load catalog
    if not dimensions_file.exists():
        console.print(f"[red]Error: Dimensions file not found: {dimensions_file}[/red]")
        raise typer.Exit(1)
    
    catalog = load_dimension_catalog(dimensions_file)
    
    # Load groups
    if not groups_path.exists():
        console.print(f"[red]Error: Groups path not found: {groups_path}[/red]")
        raise typer.Exit(1)
    
    if groups_path.is_dir():
        groups = load_groups_from_directory(groups_path)
    else:
        groups = load_groups_from_csv(groups_path)
    
    # Count stats
    total_groups = len(groups.groups)
    total_pairs = sum(len(d.pairs) for d in catalog.dimensions)
    total_prompts = total_groups * total_pairs * 7  # 7 tasks
    
    # Group counts by demography
    demography_counts: dict[str, int] = {}
    for group in groups.groups:
        dem = group.demography.value
        demography_counts[dem] = demography_counts.get(dem, 0) + 1
    
    # Display
    console.print(Panel(
        f"[bold]{catalog.name}[/bold] v{catalog.version}",
        title="Dimension Catalog",
    ))
    
    # Main stats
    main_table = Table(show_header=True, title="Overall Statistics")
    main_table.add_column("Metric", style="cyan")
    main_table.add_column("Value", justify="right")
    
    main_table.add_row("Total Groups", str(total_groups))
    main_table.add_row("Total Dimensions", str(len(catalog.dimensions)))
    main_table.add_row("Total Stereotype Pairs", str(total_pairs))
    main_table.add_row("Total Tasks", "7")
    main_table.add_row("Total Possible Prompts", str(total_prompts))
    
    console.print(main_table)
    
    # Dimension breakdown
    dim_table = Table(show_header=True, title="Pairs by Dimension")
    dim_table.add_column("Dimension", style="magenta")
    dim_table.add_column("Name", style="green")
    dim_table.add_column("Pairs", justify="right")
    
    for dim in catalog.dimensions:
        dim_table.add_row(dim.dimension.value.upper(), dim.name, str(len(dim.pairs)))
    
    console.print(dim_table)
    
    # Demography breakdown
    dem_table = Table(show_header=True, title="Groups by Demography")
    dem_table.add_column("Demography", style="yellow")
    dem_table.add_column("Count", justify="right")
    
    for dem, count in sorted(demography_counts.items()):
        dem_table.add_row(dem, str(count))
    
    console.print(dem_table)


@app.command()
def init(
    output_dir: Annotated[
        Path,
        typer.Argument(help="Directory to create sample files"),
    ] = Path("./data"),
) -> None:
    """Initialize sample dimension and group files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    groups_dir = output_dir / "groups"
    groups_dir.mkdir(exist_ok=True)
    
    sample_dimensions = '''DIMENSION_CATALOG = {
    "name": "Sample Stereotype Dimensions",
    "version": "1.0.0",
    "description": "Sample dimension catalog",
    "dimensions": [
        {
            "dimension": "sp",
            "name": "Stereotype Prevalence",
            "description": "Basic social stereotypes",
            "pairs": [
                {
                    "pair_id": "friendliness",
                    "parent_category": "Social Traits",
                    "positive_label": "friendly",
                    "negative_label": "unfriendly",
                    "description": "Basic social warmth stereotype",
                },
                {
                    "pair_id": "honesty",
                    "parent_category": "Social Traits",
                    "positive_label": "honest",
                    "negative_label": "dishonest",
                    "description": "Basic integrity stereotype",
                },
            ],
        },
        {
            "dimension": "onet",
            "name": "O*NET Occupational",
            "description": "Work skills and professional behavior",
            "pairs": [
                {
                    "pair_id": "work_ethic",
                    "parent_category": "Achievement Orientation",
                    "positive_label": "hardworking",
                    "negative_label": "lazy",
                    "description": "Simple work ethic stereotype",
                },
            ],
        },
    ],
}
'''

    dim_file = output_dir / "dimensions.py"
    dim_file.write_text(sample_dimensions)
    
    console.print(f"[green]✓[/green] Created {dim_file}")
    
    # Sample groups CSV
    sample_groups = """group_id,group_name,demography,country
indonesian,Indonesian,nationality,Indonesia
javanese,Javanese,ethnicity,Indonesia
muslim,Muslim,religion,
"""
    
    groups_file = groups_dir / "sample.csv"
    groups_file.write_text(sample_groups)
    
    console.print(f"[green]✓[/green] Created {groups_file}")
    console.print("\n[dim]Edit these files to customize your dataset.[/dim]")
    console.print("\n[cyan]File structure:[/cyan]")
    console.print("  • dimensions.py - Stereotype pairs as Python data")
    console.print("  • groups/*.csv - Demographic groups")


if __name__ == "__main__":
    app()
