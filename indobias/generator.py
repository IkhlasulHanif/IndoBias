"""Prompt generator for creating a systematic grid of prompts.

Generates prompts by combining:
1. Groups (from CSV files)
2. Stereotype pairs (from dimension catalog YAML)
3. Task templates (1-7 complexity levels)
"""

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterator

import yaml

from .data_models import (
    Demography,
    DimensionCatalog,
    GeneratedPrompt,
    Group,
    GroupCollection,
    PromptMetadata,
    StereotypeDimension,
    StereotypePair,
)
from .tasks import ALL_TASKS, TaskDefinition, get_task


def generate_prompt_id(
    group_id: str,
    dimension: str,
    pair_id: str,
    task_id: int,
) -> str:
    """Generate a unique, deterministic prompt ID."""
    content = f"{group_id}:{dimension}:{pair_id}:{task_id}"
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def generate_single_prompt(
    group: Group,
    dimension: StereotypeDimension,
    pair: StereotypePair,
    task: TaskDefinition,
) -> GeneratedPrompt:
    """Generate a single prompt from a group, stereotype pair, and task."""
    prompt_text = task.format_prompt(
        group=group.group_name,
        pos=pair.positive_label,
        neg=pair.negative_label,
        country=group.country,
        pair=pair.pair_id,
    )
    
    metadata = PromptMetadata(
        prompt_id=generate_prompt_id(group.group_id, dimension.value, pair.pair_id, task.task_id),
        demography=group.demography,
        group=group.group_name,
        group_id=group.group_id,
        dimension=dimension,
        stereotype_pair=pair.pair_id,
        positive_label=pair.positive_label,
        negative_label=pair.negative_label,
        task_id=task.task_id,
        task_complexity_level=task.complexity_level,
        task_name=task.name,
    )
    
    return GeneratedPrompt(prompt_text=prompt_text, metadata=metadata)


def generate_prompts(
    groups: GroupCollection,
    catalog: DimensionCatalog,
    task_ids: list[int] | None = None,
    demographies: list[Demography] | None = None,
    dimensions: list[StereotypeDimension] | None = None,
    use_demography_mapping: bool = True,
) -> Iterator[GeneratedPrompt]:
    """Generate all prompts by combining groups × stereotype pairs × tasks.
    
    Args:
        groups: Collection of demographic groups (from CSV)
        catalog: Catalog of stereotype dimensions/pairs (from YAML)
        task_ids: Optional list of specific task IDs (default: all 1-7)
        demographies: Optional filter for demographic categories
        dimensions: Optional filter for stereotype dimensions
        use_demography_mapping: If True, uses the demography-to-dimension mapping,
                                otherwise uses all dimensions (default: True)
    
    Yields:
        GeneratedPrompt objects for each combination
    """
    tasks = [get_task(tid) for tid in task_ids] if task_ids else ALL_TASKS
    
    for group in groups.groups:
        # Filter by demography if specified
        if demographies and group.demography not in demographies:
            continue
        
        # Determine which dimensions to use for this group
        if use_demography_mapping:
            # Use the mapped dimension for this group's demography
            try:
                group_dimension = group.get_dimension()
                dims_to_use = [d for d in catalog.dimensions if d.dimension == group_dimension]
            except ValueError:
                # Skip groups without a dimension mapping
                continue
        else:
            # Use all dimensions (or filtered ones)
            dims_to_use = catalog.dimensions
            if dimensions:
                dims_to_use = [d for d in dims_to_use if d.dimension in dimensions]
        
        # For each dimension and its pairs
        for dim_def in dims_to_use:
            for pair in dim_def.pairs:
                for task in tasks:
                    yield generate_single_prompt(group, dim_def.dimension, pair, task)


def count_prompts(
    groups: GroupCollection,
    catalog: DimensionCatalog,
    task_ids: list[int] | None = None,
    demographies: list[Demography] | None = None,
    dimensions: list[StereotypeDimension] | None = None,
) -> int:
    """Count how many prompts would be generated without actually generating them."""
    tasks_count = len(task_ids) if task_ids else 7
    
    # Count pairs in selected dimensions
    dims_to_use = catalog.dimensions
    if dimensions:
        dims_to_use = [d for d in dims_to_use if d.dimension in dimensions]
    
    total_pairs = sum(len(d.pairs) for d in dims_to_use)
    
    # Count groups
    groups_to_use = groups.groups
    if demographies:
        groups_to_use = [g for g in groups_to_use if g.demography in demographies]
    
    return len(groups_to_use) * total_pairs * tasks_count


class PromptExporter:
    """Export generated prompts to various formats."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_individual_files(
        self,
        prompts: Iterator[GeneratedPrompt],
        include_metadata: bool = True,
    ) -> int:
        """Export each prompt to its own file."""
        count = 0
        
        for prompt in prompts:
            # Create subdirectory structure: demography/dimension/task_X/
            subdir = (
                self.output_dir 
                / prompt.metadata.demography.value 
                / prompt.metadata.dimension.value
                / f"task_{prompt.metadata.task_id}"
            )
            subdir.mkdir(parents=True, exist_ok=True)
            
            filepath = subdir / prompt.metadata.to_filename()
            
            if include_metadata:
                content = "# Metadata\n"
                content += f"# prompt_id: {prompt.metadata.prompt_id}\n"
                content += f"# demography: {prompt.metadata.demography.value}\n"
                content += f"# group: {prompt.metadata.group}\n"
                content += f"# dimension: {prompt.metadata.dimension.value}\n"
                content += f"# stereotype_pair: {prompt.metadata.stereotype_pair}\n"
                content += f"# positive_label: {prompt.metadata.positive_label}\n"
                content += f"# negative_label: {prompt.metadata.negative_label}\n"
                content += f"# task_id: {prompt.metadata.task_id}\n"
                content += f"# task_name: {prompt.metadata.task_name}\n"
                content += f"# complexity_level: {prompt.metadata.task_complexity_level}\n"
                content += "\n---\n\n"
                content += prompt.prompt_text
            else:
                content = prompt.prompt_text
            
            filepath.write_text(content)
            count += 1
        
        return count
    
    def export_jsonl(
        self,
        prompts: Iterator[GeneratedPrompt],
        filename: str = "prompts.jsonl",
        organize_by_dimension: bool = True,
    ) -> int:
        """Export all prompts to JSONL file(s).
        
        Args:
            prompts: Iterator of generated prompts
            filename: Base filename for output
            organize_by_dimension: If True, creates separate files per dimension
        
        Returns:
            Total count of prompts exported
        """
        if not organize_by_dimension:
            # Single file export
            filepath = self.output_dir / filename
            count = 0
            
            with open(filepath, "w") as f:
                for prompt in prompts:
                    f.write(json.dumps(prompt.to_dict()) + "\n")
                    count += 1
            
            return count
        
        # Organize by dimension
        data_dir = self.output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        dimension_files = {}
        count = 0
        
        for prompt in prompts:
            dimension = prompt.metadata.dimension.value
            
            # Create dimension directory and file handle
            if dimension not in dimension_files:
                dim_dir = data_dir / dimension
                dim_dir.mkdir(parents=True, exist_ok=True)
                dim_path = dim_dir / filename
                dimension_files[dimension] = open(dim_path, "w")
            
            # Write to dimension-specific file
            dimension_files[dimension].write(json.dumps(prompt.to_dict()) + "\n")
            count += 1
        
        # Close all files
        for f in dimension_files.values():
            f.close()
        
        return count
    
    def export_json(
        self,
        prompts: Iterator[GeneratedPrompt],
        filename: str = "prompts.json",
    ) -> int:
        """Export all prompts to a single JSON file."""
        filepath = self.output_dir / filename
        all_prompts = [prompt.to_dict() for prompt in prompts]
        
        with open(filepath, "w") as f:
            json.dump(all_prompts, f, indent=2)
        
        return len(all_prompts)
    
    def export_yaml(
        self,
        prompts: Iterator[GeneratedPrompt],
        filename: str = "prompts.yaml",
    ) -> int:
        """Export all prompts to a single YAML file."""
        filepath = self.output_dir / filename
        all_prompts = [prompt.to_dict() for prompt in prompts]
        
        with open(filepath, "w") as f:
            yaml.dump(all_prompts, f, default_flow_style=False, allow_unicode=True)
        
        return len(all_prompts)


# =============================================================================
# LOADERS
# =============================================================================

def load_dimension_catalog(filepath: Path) -> DimensionCatalog:
    """Load stereotype dimension catalog from YAML file."""
    with open(filepath) as f:
        data = yaml.safe_load(f)
    return DimensionCatalog.model_validate(data)


def load_groups_from_csv(filepath: Path) -> GroupCollection:
    """Load groups from a CSV file.
    
    Expected CSV columns:
    - group_id: Unique identifier
    - group_name: Display name
    - notes: Optional notes (ignored)
    
    The demography is inferred from the filename:
    - ethnicity_indonesia.csv → ethnicity
    - lembaga.csv → lembaga
    - media.csv → media
    - name.csv → name
    - organisasi_bisnis.csv → organisasi_bisnis
    - political_party.csv → political_party
    - religion.csv → religion
    - university.csv → university
    - gender.csv → gender
    """
    groups = []
    
    # Infer demography from filename
    filename = filepath.stem.lower()
    
    # Map filenames to demography types
    filename_to_demography = {
        "ethnicity_indonesia": "ethnicity",
        "ethnicity": "ethnicity",
        "lembaga": "lembaga",
        "media": "media",
        "name": "name",
        "organisasi_bisnis": "organisasi_bisnis",
        "political_party": "political_party",
        "religion": "religion",
        "university": "university",
        "gender": "gender",
        "nationality": "nationality",
        "race": "race",
    }
    
    demography_str = filename_to_demography.get(filename)
    if demography_str is None:
        raise ValueError(f"Cannot infer demography from filename: {filepath.name}")
    
    demography = Demography(demography_str)
    
    # Determine country if applicable
    country = None
    if "indonesia" in filename:
        country = "Indonesia"
    
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            group = Group(
                group_id=row["group_id"].strip(),
                group_name=row["group_name"].strip(),
                demography=demography,
                country=country,
            )
            groups.append(group)
    
    return GroupCollection(
        name=filepath.stem,
        source_file=str(filepath),
        groups=groups,
    )


def load_groups_from_directory(directory: Path) -> GroupCollection:
    """Load and merge groups from all CSV files in a directory."""
    all_groups = []
    
    for csv_file in sorted(directory.glob("*.csv")):
        collection = load_groups_from_csv(csv_file)
        all_groups.extend(collection.groups)
    
    return GroupCollection(
        name=f"Groups from {directory}",
        groups=all_groups,
    )


def load_groups_from_yaml(filepath: Path) -> GroupCollection:
    """Load groups from a YAML file (alternative to CSV)."""
    with open(filepath) as f:
        data = yaml.safe_load(f)
    return GroupCollection.model_validate(data)

