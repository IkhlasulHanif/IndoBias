"""Simple Typer CLI for the IndoBias QA package."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .config import (
    DEFAULT_RAW_RESULTS_DIR,
    DIMENSIONS_FILE,
    GROUPS_DIR,
    VALID_DEMOGRAPHIES,
    VALID_DIMENSIONS,
)
from .data_models import Demography, Group, GroupCollection, StereotypeDimension
from .generator import (
    PromptExporter,
    count_prompts,
    generate_prompts,
    load_dimension_catalog,
    load_groups_from_csv,
    load_groups_from_directory,
)
from .models.openai_batch import (
    DEFAULT_BATCH_ROOT,
    DEFAULT_MODELS as DEFAULT_OPENAI_BATCH_MODELS,
    download_outputs as openai_batch_download_outputs,
    export_outputs as openai_batch_export_outputs,
    export_outputs_from_roots as openai_batch_export_outputs_from_roots,
    prepare_batches as openai_batch_prepare_batches,
    refresh_status as openai_batch_refresh_status,
    submit_batches as openai_batch_submit_batches,
    watch_status as openai_batch_watch_status,
)
from .tasks import ALL_TASKS, get_task

cli = typer.Typer(
    name="indobias",
    help="Task complexity framework for testing stereotype expression in LLMs",
    add_completion=False,
)
console = Console()
TASK_ID_RANGE = range(1, 8)
EXPORTERS = {
    "xlsx": lambda exporter, prompts, model_name, no_metadata: exporter.export_xlsx_by_demography_and_file(
        prompts,
        model_name=model_name,
    ),
    "files": lambda exporter, prompts, model_name, no_metadata: exporter.export_individual_files(
        prompts,
        include_metadata=not no_metadata,
    ),
    "jsonl": lambda exporter, prompts, model_name, no_metadata: exporter.export_jsonl(
        prompts,
        organize_by_dimension=True,
    ),
    "json": lambda exporter, prompts, model_name, no_metadata: exporter.export_json(prompts),
}


def fail(message: str, *, valid_options: tuple[str, ...] | None = None) -> None:
    console.print(f"[red]Error:[/red] {message}")
    if valid_options:
        console.print(f"Valid options: {', '.join(valid_options)}")
    raise typer.Exit(1)


def require_path(path: Path, label: str) -> Path:
    if not path.exists():
        fail(f"{label} not found: {path}")
    return path


def load_catalog(dimensions_file: Path):
    return load_dimension_catalog(require_path(dimensions_file, "Dimensions file"))


def load_groups(groups_path: Path) -> GroupCollection:
    path = require_path(groups_path, "Groups path")
    if path.is_dir():
        return load_groups_from_directory(path)
    return load_groups_from_csv(path)


def parse_task_ids(value: str | None) -> list[int] | None:
    if not value or value.lower() == "all":
        return None
    try:
        task_ids = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        fail(str(exc))
    invalid = [task_id for task_id in task_ids if task_id not in TASK_ID_RANGE]
    if invalid:
        fail(f"Invalid task ID: {invalid[0]}")
    return task_ids


def parse_demographies(value: str | None) -> list[Demography] | None:
    if not value:
        return None
    try:
        return [Demography(value.lower())]
    except ValueError:
        fail(f"Invalid demography '{value}'", valid_options=VALID_DEMOGRAPHIES)


def parse_dimension(value: str | None) -> list[StereotypeDimension] | None:
    if not value:
        return None
    try:
        return [StereotypeDimension(value.lower())]
    except ValueError:
        fail(f"Invalid dimension '{value}'", valid_options=VALID_DIMENSIONS)


def get_groups_for_preview(groups: GroupCollection, group_id: str | None) -> list[Group]:
    if not group_id:
        return groups.groups[:3]
    group = groups.get_group(group_id)
    if not group:
        fail(f"Group '{group_id}' not found")
    return [group]


def group_by_demography(groups: list[Group]) -> dict[str, list[Group]]:
    grouped: dict[str, list[Group]] = {}
    for group in groups:
        grouped.setdefault(group.demography.value, []).append(group)
    return grouped


def render_task_table() -> None:
    table = Table(title="Task Complexity Levels", show_header=True)
    table.add_column("ID", style="cyan", justify="center")
    table.add_column("Name", style="green")
    table.add_column("Level", justify="center")
    table.add_column("Rationale", style="dim", max_width=50)

    for task in ALL_TASKS:
        rationale = task.rationale.replace("\n", " ").strip()[:50] + "..."
        table.add_row(str(task.task_id), task.name, str(task.complexity_level), rationale)

    console.print(table)


def render_dimensions(dimensions_file: Path) -> None:
    catalog = load_catalog(dimensions_file)
    console.print(
        Panel(
            f"[bold]{catalog.name}[/bold] v{catalog.version}\n"
            f"[dim]{catalog.description or 'No description'}[/dim]",
            title="Dimension Catalog",
        )
    )

    for dim in catalog.dimensions:
        table = Table(title=f"{dim.name} ({dim.dimension.value.upper()})", show_header=True)
        table.add_column("Pair ID", style="cyan")
        table.add_column("Positive", style="green")
        table.add_column("Negative", style="red")
        table.add_column("Description", style="dim", max_width=30)
        for pair in dim.pairs:
            table.add_row(pair.pair_id, pair.positive_label, pair.negative_label, pair.description or "")
        console.print(table)
        console.print()


def render_groups(groups_path: Path, demography: str | None) -> None:
    groups = load_groups(groups_path)
    filtered = groups.groups if not demography else groups.get_groups_by_demography(parse_demographies(demography)[0])

    for name, demography_groups in sorted(group_by_demography(filtered).items()):
        table = Table(title=f"{name.upper()} ({len(demography_groups)} groups)", show_header=True)
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Country", style="dim")
        for group in demography_groups:
            table.add_row(group.group_id, group.group_name, group.country or "")
        console.print(table)
        console.print()


def render_stats(dimensions_file: Path, groups_path: Path) -> None:
    catalog = load_catalog(dimensions_file)
    groups = load_groups(groups_path)
    total_groups = len(groups.groups)
    total_pairs = sum(len(dim.pairs) for dim in catalog.dimensions)

    console.print(Panel(f"[bold]{catalog.name}[/bold] v{catalog.version}", title="Dimension Catalog"))

    main_table = Table(show_header=True, title="Overall Statistics")
    main_table.add_column("Metric", style="cyan")
    main_table.add_column("Value", justify="right")
    main_table.add_row("Total Groups", str(total_groups))
    main_table.add_row("Total Dimensions", str(len(catalog.dimensions)))
    main_table.add_row("Total Stereotype Pairs", str(total_pairs))
    main_table.add_row("Total Tasks", "7")
    main_table.add_row("Total Possible Prompts", str(total_groups * total_pairs * 7))
    console.print(main_table)

    dim_table = Table(show_header=True, title="Pairs by Dimension")
    dim_table.add_column("Dimension", style="magenta")
    dim_table.add_column("Name", style="green")
    dim_table.add_column("Pairs", justify="right")
    for dim in catalog.dimensions:
        dim_table.add_row(dim.dimension.value.upper(), dim.name, str(len(dim.pairs)))
    console.print(dim_table)

    dem_table = Table(show_header=True, title="Groups by Demography")
    dem_table.add_column("Demography", style="yellow")
    dem_table.add_column("Count", justify="right")
    for demography, demography_groups in sorted(group_by_demography(groups.groups).items()):
        dem_table.add_row(demography, str(len(demography_groups)))
    console.print(dem_table)


def render_preview(
    dimensions_file: Path,
    groups_path: Path,
    task_id: int,
    group_id: str | None,
    dimension: str | None,
) -> None:
    catalog = load_catalog(dimensions_file)
    groups = load_groups(groups_path)
    try:
        task = get_task(task_id)
    except ValueError as exc:
        fail(str(exc))

    dims_to_show = catalog.dimensions
    dimension_filter = parse_dimension(dimension)
    if dimension_filter:
        dims_to_show = [dim for dim in dims_to_show if dim.dimension in dimension_filter]

    console.print(f"[cyan]Previewing Task {task_id}: {task.name}[/cyan]\n")
    for group in get_groups_for_preview(groups, group_id):
        for dim in dims_to_show[:1]:
            for pair in dim.pairs[:2]:
                prompt = task.format_prompt(
                    group=group.group_name,
                    pos=pair.positive_label,
                    neg=pair.negative_label,
                    country=group.country,
                )
                console.print(
                    Panel(
                        prompt,
                        title=f"[green]{group.group_name}[/green] | [magenta]{dim.dimension.value.upper()}[/magenta] | {pair.pair_id}",
                        border_style="dim",
                    )
                )
                console.print()


@cli.command(name="generate")
def generate(
    dimensions_file: Annotated[Path, typer.Argument(help="Path to dimensions catalog Python file")] = DIMENSIONS_FILE,
    groups_path: Annotated[Path, typer.Option("--groups", "-g", help="Path to groups CSV file or directory")] = GROUPS_DIR,
    output_dir: Annotated[Path, typer.Option("--output", "-o", help="Output directory for generated prompts")] = DEFAULT_RAW_RESULTS_DIR,
    format: Annotated[str, typer.Option("--format", "-f", help="Output format: xlsx, files, jsonl, json")] = "xlsx",
    model_name: Annotated[str, typer.Option("--model-name", help="Model name used to partition XLSX exports")] = "prompts",
    tasks: Annotated[str | None, typer.Option("--tasks", "-t", help="Comma-separated task IDs (e.g., '1,3,5') or 'all'")] = None,
    demography: Annotated[str | None, typer.Option("--demography", "-d", help="Filter by demography")] = None,
    dimension: Annotated[str | None, typer.Option("--dimension", "-D", help="Filter by dimension")] = None,
    no_metadata: Annotated[bool, typer.Option("--no-metadata", help="Exclude metadata from individual files")] = False,
) -> None:
    """Generate prompts from dimensions and groups."""
    catalog = load_catalog(dimensions_file)
    groups = load_groups(groups_path)
    task_ids = parse_task_ids(tasks)
    demographies = parse_demographies(demography)
    dimensions_filter = parse_dimension(dimension)

    total = count_prompts(groups, catalog, task_ids, demographies, dimensions_filter)
    console.print(f"Total prompts to generate: {total}")
    if total == 0:
        console.print("[yellow]No prompts to generate with the specified filters.[/yellow]")
        raise typer.Exit(0)

    export = EXPORTERS.get(format)
    if export is None:
        fail(f"Unknown format '{format}'")

    exporter = PromptExporter(output_dir)
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        progress.add_task("Generating prompts...", total=None)
        prompts = generate_prompts(groups, catalog, task_ids, demographies, dimensions_filter)
        count = export(exporter, prompts, model_name, no_metadata)

    console.print(f"[green]✓[/green] Generated {count} prompts to {output_dir}")


@cli.command(name="list-tasks")
def list_tasks() -> None:
    """List all available task templates with their complexity levels."""
    render_task_table()


@cli.command(name="list-dimensions")
def list_dimensions(
    dimensions_file: Annotated[Path, typer.Argument(help="Path to dimensions catalog Python file")] = DIMENSIONS_FILE,
) -> None:
    """List all stereotype dimensions and their pairs."""
    render_dimensions(dimensions_file)


@cli.command(name="list-groups")
def list_groups(
    groups_path: Annotated[Path, typer.Argument(help="Path to groups CSV file or directory")] = GROUPS_DIR,
    demography: Annotated[str | None, typer.Option("--demography", "-d", help="Filter by demography")] = None,
) -> None:
    """List all demographic groups from CSV files."""
    render_groups(groups_path, demography)


@cli.command(name="show-task")
def show_task(task_id: Annotated[int, typer.Argument(help="Task ID (1-7)")]) -> None:
    """Show detailed information about a specific task."""
    try:
        task = get_task(task_id)
    except ValueError as exc:
        fail(str(exc))

    console.print(
        Panel(
            f"[bold cyan]{task.name}[/bold cyan]\n\n"
            f"[yellow]Complexity Level:[/yellow] {task.complexity_level}/7\n\n"
            f"[yellow]Rationale:[/yellow]\n{task.rationale}\n\n"
            f"[yellow]Template:[/yellow]\n[dim]{task.template}[/dim]",
            title=f"Task {task.task_id}",
            expand=False,
        )
    )


@cli.command(name="preview")
def preview(
    dimensions_file: Annotated[Path, typer.Option("--dimensions", help="Path to dimensions catalog Python file")] = DIMENSIONS_FILE,
    groups_path: Annotated[Path, typer.Option("--groups", "-g", help="Path to groups CSV file or directory")] = GROUPS_DIR,
    task_id: Annotated[int, typer.Option("--task", "-t", help="Task ID to preview (1-7)")] = 1,
    group_id: Annotated[str | None, typer.Option("--group", help="Specific group ID to preview")] = None,
    dimension: Annotated[str | None, typer.Option("--dimension", "-D", help="Filter by dimension")] = None,
) -> None:
    """Preview generated prompts without saving to files."""
    render_preview(dimensions_file, groups_path, task_id, group_id, dimension)


@cli.command(name="stats")
def stats(
    dimensions_file: Annotated[Path, typer.Option("--dimensions", help="Path to dimensions catalog Python file")] = DIMENSIONS_FILE,
    groups_path: Annotated[Path, typer.Option("--groups", "-g", help="Path to groups CSV file or directory")] = GROUPS_DIR,
) -> None:
    """Show statistics about dimensions and groups."""
    render_stats(dimensions_file, groups_path)


@cli.command(name="init")
def init(
    output_dir: Annotated[Path, typer.Argument(help="Directory to create sample files")] = Path("./data"),
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
    (output_dir / "dimensions.py").write_text(sample_dimensions)
    console.print(f"[green]✓[/green] Created {output_dir / 'dimensions.py'}")

    sample_groups = """group_id,group_name,demography,country
indonesian,Indonesian,nationality,Indonesia
javanese,Javanese,ethnicity,Indonesia
muslim,Muslim,religion,
"""
    (groups_dir / "sample.csv").write_text(sample_groups)
    console.print(f"[green]✓[/green] Created {groups_dir / 'sample.csv'}")
    console.print("\n[dim]Edit these files to customize your dataset.[/dim]")
    console.print("\n[cyan]File structure:[/cyan]")
    console.print("  • dimensions.py - Stereotype pairs as Python data")
    console.print("  • groups/*.csv - Demographic groups")


@cli.command(name="openai-batch-prepare")
def openai_batch_prepare(
    dimensions_file: Annotated[Path, typer.Option("--dimensions", help="Path to dimensions catalog Python file")] = DIMENSIONS_FILE,
    groups_path: Annotated[Path, typer.Option("--groups", "-g", help="Path to groups CSV file or directory")] = GROUPS_DIR,
    tasks: Annotated[str | None, typer.Option("--tasks", "-t", help="Comma-separated task IDs or 'all'")] = None,
    demography: Annotated[str | None, typer.Option("--demography", "-d", help="Filter by demography")] = None,
    dimension: Annotated[str | None, typer.Option("--dimension", "-D", help="Filter by dimension")] = None,
    models: Annotated[list[str], typer.Option("--model", help="OpenAI model to include; repeat for multiple models")] = list(DEFAULT_OPENAI_BATCH_MODELS),
    batch_root: Annotated[Path, typer.Option("--batch-root", help="Directory for batch request and output files")] = DEFAULT_BATCH_ROOT,
    max_output_tokens: Annotated[int, typer.Option("--max-output-tokens", help="max_output_tokens sent to the OpenAI Responses API")] = 1024,
    num_generations: Annotated[int, typer.Option("--num-generations", help="Number of repeated stochastic requests per prompt")] = 1,
    generation_start: Annotated[int, typer.Option("--generation-start", help="First generation_index to assign when splitting repeated batches")] = 0,
    temperature: Annotated[float | None, typer.Option("--temperature", help="Sampling temperature sent to the OpenAI Responses API")] = None,
    top_p: Annotated[float | None, typer.Option("--top-p", help="Nucleus sampling parameter sent to the OpenAI Responses API")] = None,
) -> None:
    openai_batch_prepare_batches(
        batch_root=batch_root,
        dimensions_file=dimensions_file,
        groups_path=groups_path,
        tasks=tasks,
        demography=demography,
        dimension=dimension,
        models=models,
        max_output_tokens=max_output_tokens,
        num_generations=num_generations,
        generation_start=generation_start,
        temperature=temperature,
        top_p=top_p,
    )


@cli.command(name="openai-batch-submit")
def openai_batch_submit(
    batch_root: Annotated[Path, typer.Option("--batch-root", help="Directory for batch request and output files")] = DEFAULT_BATCH_ROOT,
) -> None:
    openai_batch_submit_batches(batch_root=batch_root)


@cli.command(name="openai-batch-status")
def openai_batch_status(
    batch_root: Annotated[Path, typer.Option("--batch-root", help="Directory for batch request and output files")] = DEFAULT_BATCH_ROOT,
    watch: Annotated[bool, typer.Option("--watch", help="Poll until all batches reach a terminal state")] = False,
    interval_seconds: Annotated[int, typer.Option("--interval-seconds", help="Polling interval in seconds for --watch")] = 30,
) -> None:
    if watch:
        openai_batch_watch_status(batch_root=batch_root, interval_seconds=interval_seconds)
        return
    openai_batch_refresh_status(batch_root=batch_root)


@cli.command(name="openai-batch-download")
def openai_batch_download(
    batch_root: Annotated[Path, typer.Option("--batch-root", help="Directory for batch request and output files")] = DEFAULT_BATCH_ROOT,
) -> None:
    openai_batch_download_outputs(batch_root=batch_root)


@cli.command(name="openai-batch-export")
def openai_batch_export(
    batch_root: Annotated[Path, typer.Option("--batch-root", help="Directory for batch request and output files")] = DEFAULT_BATCH_ROOT,
    output_dir: Annotated[Path, typer.Option("--output", "-o", help="Directory for final Excel outputs")] = DEFAULT_RAW_RESULTS_DIR,
) -> None:
    openai_batch_export_outputs(batch_root=batch_root, raw_output_dir=output_dir)


@cli.command(name="openai-batch-export-many")
def openai_batch_export_many(
    batch_roots: Annotated[list[Path], typer.Argument(help="Batch roots to merge/export")],
    output_dir: Annotated[Path, typer.Option("--output", "-o", help="Directory for final Excel outputs")] = DEFAULT_RAW_RESULTS_DIR,
) -> None:
    openai_batch_export_outputs_from_roots(
        batch_roots=batch_roots,
        raw_output_dir=output_dir,
    )


@cli.command(name="openai-batch-prepare-submit")
def openai_batch_prepare_submit(
    dimensions_file: Annotated[Path, typer.Option("--dimensions", help="Path to dimensions catalog Python file")] = DIMENSIONS_FILE,
    groups_path: Annotated[Path, typer.Option("--groups", "-g", help="Path to groups CSV file or directory")] = GROUPS_DIR,
    tasks: Annotated[str | None, typer.Option("--tasks", "-t", help="Comma-separated task IDs or 'all'")] = None,
    demography: Annotated[str | None, typer.Option("--demography", "-d", help="Filter by demography")] = None,
    dimension: Annotated[str | None, typer.Option("--dimension", "-D", help="Filter by dimension")] = None,
    models: Annotated[list[str], typer.Option("--model", help="OpenAI model to include; repeat for multiple models")] = list(DEFAULT_OPENAI_BATCH_MODELS),
    batch_root: Annotated[Path, typer.Option("--batch-root", help="Directory for batch request and output files")] = DEFAULT_BATCH_ROOT,
    max_output_tokens: Annotated[int, typer.Option("--max-output-tokens", help="max_output_tokens sent to the OpenAI Responses API")] = 1024,
    num_generations: Annotated[int, typer.Option("--num-generations", help="Number of repeated stochastic requests per prompt")] = 1,
    generation_start: Annotated[int, typer.Option("--generation-start", help="First generation_index to assign when splitting repeated batches")] = 0,
    temperature: Annotated[float | None, typer.Option("--temperature", help="Sampling temperature sent to the OpenAI Responses API")] = None,
    top_p: Annotated[float | None, typer.Option("--top-p", help="Nucleus sampling parameter sent to the OpenAI Responses API")] = None,
) -> None:
    openai_batch_prepare(
        dimensions_file=dimensions_file,
        groups_path=groups_path,
        tasks=tasks,
        demography=demography,
        dimension=dimension,
        models=models,
        batch_root=batch_root,
        max_output_tokens=max_output_tokens,
        num_generations=num_generations,
        generation_start=generation_start,
        temperature=temperature,
        top_p=top_p,
    )
    openai_batch_submit(batch_root=batch_root)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
