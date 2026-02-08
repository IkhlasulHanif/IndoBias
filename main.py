"""Main script: Generate prompts and run inference for all demography groups.

Usage:
    # Run full pipeline for all groups
    python main.py
    
    # Run for specific groups only
    python main.py --groups ethnicity_indonesia gender religion
    
    # Use a different model
    python main.py --model qwen --model-id Qwen/Qwen3-3B-Instruct
    
    # Only generate prompts (skip inference)
    python main.py --generate-only
    
    # Only run inference (skip generation)
    python main.py --skip-generation
    
    # List available groups
    python main.py --list-groups
    
    # List available models
    python main.py --list-models

Features:
    - Uses indobias.generator for prompt generation
    - Uses indobias.data_models for data structures
    - Uses indobias.tasks for task templates
    - Supports multiple model backends (qwen, openai, anthropic, gemini)
    - Progress bars with rich
    - Organized output structure
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

from indobias.generator import (
    generate_prompts,
    load_dimension_catalog,
    load_groups_from_csv,
    load_groups_from_directory,
    PromptExporter,
)

console = Console()


def load_model(model_name: str, model_id: str, **kwargs):
    """
    Load a model based on the model name.
    
    Args:
        model_name: Name of the model ('qwen', 'openai', 'anthropic', etc.)
        model_id: Model identifier/path
        **kwargs: Additional arguments for model initialization
    
    Returns:
        Model instance with generate() method
    """
    if model_name == "qwen":
        from indobias.models import QwenModel
        return QwenModel(model_name=model_id, **kwargs)
    elif model_name == "openai":
        from indobias.models.openai import OpenAIModel
        return OpenAIModel(model_name=model_id, **kwargs)
    elif model_name == "anthropic":
        from indobias.models.anthropic import AnthropicModel
        return AnthropicModel(model_name=model_id, **kwargs)
    elif model_name == "gemini":
        from indobias.models.gemini import GeminiModel
        return GeminiModel(model_name=model_id, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}. Available: qwen, openai, anthropic, gemini")


def list_available_groups():
    """List all available group files."""
    groups_dir = Path("data/groups")
    if not groups_dir.exists():
        console.print(f"[red]❌ Groups directory not found: {groups_dir}[/red]")
        return
    
    csv_files = sorted(groups_dir.glob("*.csv"))
    
    console.print("\n[cyan]Available Group Files:[/cyan]")
    if not csv_files:
        console.print("  No group files found.")
    else:
        for i, csv_file in enumerate(csv_files, 1):
            console.print(f"  {i:2d}. {csv_file.stem}")
    
    console.print("\n[dim]Usage: python main.py --groups ethnicity_indonesia gender[/dim]\n")


def list_available_models():
    """List all available models."""
    console.print("\n[cyan]Available Models:[/cyan]")
    console.print("  1. qwen       [green]✓[/green] Implemented")
    console.print("  2. openai     [red]✗[/red] Not implemented yet")
    console.print("  3. anthropic  [red]✗[/red] Not implemented yet")
    console.print("  4. gemini     [red]✗[/red] Not implemented yet")
    console.print("\n[dim]To add a new model, create indobias/models/<model_name>.py[/dim]\n")


def generate_all_prompts(groups_to_use: Optional[list[str]] = None):
    """
    Generate prompts for specified groups or all groups.
    
    Args:
        groups_to_use: List of group file names (without .csv) or None for all
    
    Returns:
        True if successful, False otherwise
    """
    console.print("\n[bold cyan]═" * 40 + "[/bold cyan]")
    console.print("[bold cyan]STEP 1: GENERATING PROMPTS[/bold cyan]")
    console.print("[bold cyan]═" * 40 + "[/bold cyan]\n")
    
    groups_dir = Path("data/groups")
    dimensions_file = Path("data/dimensions.yaml")
    output_dir = Path("output")
    
    # Load dimension catalog
    console.print("[cyan]Loading dimensions...[/cyan]")
    catalog = load_dimension_catalog(dimensions_file)
    console.print(f"[green]✓[/green] Loaded {len(catalog.dimensions)} dimensions\n")
    
    # Load groups
    console.print("[cyan]Loading groups...[/cyan]")
    
    if groups_to_use:
        # Load specific groups
        all_groups_list = []
        for group_name in groups_to_use:
            csv_path = groups_dir / f"{group_name}.csv"
            if csv_path.exists():
                groups = load_groups_from_csv(csv_path)
                all_groups_list.extend(groups.groups)
                console.print(f"  • {group_name}: {len(groups.groups)} groups")
            else:
                console.print(f"  [yellow]⚠ {group_name}.csv not found[/yellow]")
        
        if not all_groups_list:
            console.print("[red]❌ No valid groups found[/red]")
            return False
        
        from indobias.data_models import GroupCollection
        groups_collection = GroupCollection(name="Selected Groups", groups=all_groups_list)
    else:
        # Load all groups
        groups_collection = load_groups_from_directory(groups_dir)
        console.print(f"  • All groups from directory")
    
    console.print(f"[green]✓[/green] Loaded {len(groups_collection.groups)} total groups\n")
    
    # Generate prompts
    console.print("[cyan]Generating prompts...[/cyan]")
    exporter = PromptExporter(output_dir)
    prompts = generate_prompts(groups_collection, catalog)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Exporting prompts...", total=None)
        count = exporter.export_jsonl(prompts, organize_by_dimension=True)
        progress.update(task, completed=True)
    
    console.print(f"[green]✓[/green] Generated {count} prompts to {output_dir}/\n")
    return True


def run_inference_pipeline(model_name: str = "qwen", model_id: str = "Qwen/Qwen3-8B"):
    """
    Run inference on all generated prompts.
    
    Args:
        model_name: Name of the model to use
        model_id: Model identifier/path
    
    Returns:
        True if successful, False otherwise
    """
    console.print("\n[bold cyan]═" * 40 + "[/bold cyan]")
    console.print("[bold cyan]STEP 2: RUNNING INFERENCE[/bold cyan]")
    console.print("[bold cyan]═" * 40 + "[/bold cyan]\n")
    
    output_dir = Path("output")
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Find and load all prompts
    console.print("[cyan]Loading prompts...[/cyan]")
    data_dir = output_dir / "data"
    
    if not data_dir.exists():
        console.print("[red]❌ No prompts found. Run generation first.[/red]")
        return False
    
    prompt_files = sorted(data_dir.glob("*/prompts.jsonl"))
    
    if not prompt_files:
        console.print("[red]❌ No prompt files found[/red]")
        return False
    
    console.print(f"[green]✓[/green] Found {len(prompt_files)} dimension file(s)")
    
    # Load all prompts
    all_prompts = []
    for pf in prompt_files:
        with open(pf, 'r') as f:
            for line in f:
                all_prompts.append(json.loads(line))
    
    console.print(f"[green]✓[/green] Loaded {len(all_prompts)} total prompts\n")
    
    # Initialize model
    console.print(f"[cyan]Initializing {model_name} model...[/cyan]")
    console.print(f"  Model ID: {model_id}")
    
    try:
        if model_name == "qwen":
            model = load_model(model_name, model_id, max_length=100, temperature=0.7)
        else:
            model = load_model(model_name, model_id)
    except Exception as e:
        console.print(f"[red]❌ Failed to load model: {e}[/red]")
        return False
    
    console.print()
    
    # Create output directory
    timestamp = datetime.now().strftime("%m-%d_%H%M%S")
    model_short = model_id.split('/')[-1].replace('.', '_').replace('-', '_').lower()
    run_dir = models_dir / f"{model_name}_{model_short}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[cyan]Output directory:[/cyan] {run_dir.relative_to(output_dir)}\n")
    
    # Organize prompts by demography → dimension → task
    prompts_by_structure = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for prompt_data in all_prompts:
        meta = prompt_data["metadata"]
        demography = meta["demography"]
        dimension = meta["dimension"]
        task_id = meta["task_id"]
        prompts_by_structure[demography][dimension][task_id].append(prompt_data)
    
    # Run inference
    total_processed = 0
    result_id = 1
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        
        main_task = progress.add_task(
            "[cyan]Processing all prompts...", 
            total=len(all_prompts)
        )
        
        for demography in sorted(prompts_by_structure.keys()):
            demography_dir = run_dir / demography
            demography_dir.mkdir(exist_ok=True)
            
            for dimension in sorted(prompts_by_structure[demography].keys()):
                dim_dir = demography_dir / dimension
                dim_dir.mkdir(exist_ok=True)
                
                for task_id in sorted(prompts_by_structure[demography][dimension].keys()):
                    task_prompts = prompts_by_structure[demography][dimension][task_id]
                    results_file = dim_dir / f"task_{task_id}_results.jsonl"
                    
                    with open(results_file, 'w') as out_f:
                        for prompt_data in task_prompts:
                            prompt_text = prompt_data["prompt"]
                            metadata = prompt_data["metadata"]
                            
                            try:
                                response = model.generate(prompt_text, max_new_tokens=50)
                                
                                result = {
                                    "id": result_id,
                                    "demography": demography,
                                    "prompt": prompt_text,
                                    "response": response,
                                    "metadata": metadata,
                                    "model": getattr(model, 'model_name', model_id),
                                    "device": getattr(model, 'device', 'unknown'),
                                    "timestamp": datetime.now().isoformat()
                                }
                                out_f.write(json.dumps(result) + "\n")
                                
                            except Exception as e:
                                result = {
                                    "id": result_id,
                                    "demography": demography,
                                    "prompt": prompt_text,
                                    "response": None,
                                    "error": str(e),
                                    "metadata": metadata,
                                    "model": getattr(model, 'model_name', model_id),
                                    "device": getattr(model, 'device', 'unknown'),
                                    "timestamp": datetime.now().isoformat()
                                }
                                out_f.write(json.dumps(result) + "\n")
                            
                            total_processed += 1
                            result_id += 1
                            progress.update(main_task, advance=1)
    
    console.print(f"\n[green]✓[/green] Processed {total_processed} prompts")
    console.print(f"[green]✓[/green] Results saved to: {run_dir.relative_to(output_dir)}/\n")
    
    return True


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="IndoBias: Generate prompts and run inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run for all groups (default)
  python main.py
  
  # Run for specific groups
  python main.py --groups ethnicity_indonesia gender religion
  
    # Use different model
    python main.py --model qwen --model-id Qwen/Qwen3-3B-Instruct
  
  # Skip prompt generation
  python main.py --skip-generation
  
  # Only generate prompts
  python main.py --generate-only
        """
    )
    
    parser.add_argument(
        '--groups',
        nargs='+',
        default=None,
        help='Group file names (without .csv). Default: all groups'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='qwen',
        choices=['qwen', 'openai', 'anthropic', 'gemini'],
        help='Model to use (default: qwen)'
    )
    
    parser.add_argument(
        '--model-id',
        type=str,
        default='Qwen/Qwen3-8B',
        help='Model identifier (default: Qwen/Qwen3-8B)'
    )
    
    parser.add_argument(
        '--skip-generation',
        action='store_true',
        help='Skip prompt generation, use existing prompts'
    )
    
    parser.add_argument(
        '--generate-only',
        action='store_true',
        help='Only generate prompts, skip inference'
    )
    
    parser.add_argument(
        '--list-groups',
        action='store_true',
        help='List available group files'
    )
    
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List available models'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Handle list commands
    if args.list_groups:
        list_available_groups()
        return 0
    
    if args.list_models:
        list_available_models()
        return 0
    
    # Display header
    console.print("\n[bold]╔" + "═" * 78 + "╗[/bold]")
    console.print("[bold]║" + " " * 78 + "║[/bold]")
    console.print("[bold]║" + "IndoBias: Full Pipeline".center(78) + "║[/bold]")
    console.print("[bold]║" + " " * 78 + "║[/bold]")
    console.print("[bold]╚" + "═" * 78 + "╝[/bold]")
    
    # Display configuration
    console.print("\n[cyan]Configuration:[/cyan]")
    console.print(f"  Groups: {', '.join(args.groups) if args.groups else 'ALL'}")
    console.print(f"  Model: {args.model}")
    console.print(f"  Model ID: {args.model_id}")
    
    start_time = datetime.now()
    
    # Step 1: Generate prompts
    if not args.skip_generation:
        if not generate_all_prompts(args.groups):
            console.print("\n[red]❌ Prompt generation failed[/red]")
            return 1
    else:
        console.print("\n[yellow]Skipping prompt generation[/yellow]")
    
    # Step 2: Run inference
    if not args.generate_only:
        if not run_inference_pipeline(args.model, args.model_id):
            console.print("\n[red]❌ Inference failed[/red]")
            return 1
    else:
        console.print("\n[yellow]Skipping inference[/yellow]")
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    console.print("\n[bold green]═" * 40 + "[/bold green]")
    console.print("[bold green]🎉 PIPELINE COMPLETED![/bold green]")
    console.print("[bold green]═" * 40 + "[/bold green]")
    console.print(f"  Duration: {duration}")
    console.print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
