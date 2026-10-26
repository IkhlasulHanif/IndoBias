"""Prompt generator for creating a systematic grid of prompts."""

import csv
import hashlib
import json
from pathlib import Path
import runpy
from typing import Iterator

import pandas as pd

from .config import infer_group_metadata
from .data_models import (
    Demography,
    DimensionDefinition,
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


def normalize_model_name(model_name: str) -> str:
    """Normalize a model name for use in directory and file metadata."""
    return (
        model_name.strip()
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


def export_rows_by_demography_and_file(
    output_dir: Path,
    rows: list[dict[str, object]],
    model_name: str,
) -> None:
    """Export rows into one XLSX file per source CSV under demography directories."""
    grouped_rows: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    normalized_model_name = normalize_model_name(model_name)

    for row in rows:
        demography = str(row["demography"])
        source_file = str(row.get("source_file") or demography)
        key = (demography, normalized_model_name, source_file)
        grouped_rows.setdefault(key, []).append(row)

    base_output_dir = Path(output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    for (demography, normalized_name, source_file), grouped in grouped_rows.items():
        demography_dir = base_output_dir / demography
        demography_dir.mkdir(parents=True, exist_ok=True)
        filepath = demography_dir / f"{normalized_name}__{source_file}.xlsx"
        pd.DataFrame(grouped).to_excel(filepath, index=False)


def generate_single_prompt(
    group: Group,
    dimension_def: DimensionDefinition,
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
        prompt_id=generate_prompt_id(
            group.group_id,
            dimension_def.dimension.value,
            pair.pair_id,
            task.task_id,
        ),
        demography=group.demography,
        source_file=group.source_file,
        group=group.group_name,
        group_id=group.group_id,
        dimension=dimension_def.dimension,
        dimension_name=dimension_def.name,
        stereotype_pair=pair.pair_id,
        parent_category=pair.parent_category,
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
        catalog: Catalog of stereotype dimensions/pairs
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
                if dimensions and group_dimension not in dimensions:
                    continue
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
                    yield generate_single_prompt(group, dim_def, pair, task)


def count_prompts(
    groups: GroupCollection,
    catalog: DimensionCatalog,
    task_ids: list[int] | None = None,
    demographies: list[Demography] | None = None,
    dimensions: list[StereotypeDimension] | None = None,
    use_demography_mapping: bool = True,
) -> int:
    """Count how many prompts would be generated without actually generating them."""
    tasks_count = len(task_ids) if task_ids else 7

    groups_to_use = groups.groups
    if demographies:
        groups_to_use = [g for g in groups_to_use if g.demography in demographies]

    if not use_demography_mapping:
        dims_to_use = catalog.dimensions
        if dimensions:
            dims_to_use = [d for d in dims_to_use if d.dimension in dimensions]
        return len(groups_to_use) * sum(len(d.pairs) for d in dims_to_use) * tasks_count

    total = 0
    for group in groups_to_use:
        try:
            group_dimension = group.get_dimension()
        except ValueError:
            continue
        if dimensions and group_dimension not in dimensions:
            continue
        dimension_def = catalog.get_dimension(group_dimension)
        if dimension_def is None:
            continue
        total += len(dimension_def.pairs) * tasks_count
    return total


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
                content += f"# parent_category: {prompt.metadata.parent_category}\n"
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

    def export_xlsx_by_demography_and_file(
        self,
        prompts: Iterator[GeneratedPrompt],
        model_name: str = "prompts",
    ) -> int:
        """Export prompts into one XLSX file per source CSV under demography directories."""
        rows: list[dict[str, object]] = []
        count = 0

        for prompt in prompts:
            metadata = prompt.metadata
            demography = metadata.demography.value
            source_file = metadata.source_file or demography
            rows.append(
                {
                    "model_name": model_name,
                    "prompt_id": metadata.prompt_id,
                    "demography": demography,
                    "source_file": source_file,
                    "group_id": metadata.group_id,
                    "group": metadata.group,
                    "dimension": metadata.dimension.value,
                    "dimension_name": metadata.dimension_name,
                    "stereotype_pair": metadata.stereotype_pair,
                    "parent_category": metadata.parent_category,
                    "positive_label": metadata.positive_label,
                    "negative_label": metadata.negative_label,
                    "task_id": metadata.task_id,
                    "task_name": metadata.task_name,
                    "task_complexity_level": metadata.task_complexity_level,
                    "prompt": prompt.prompt_text,
                }
            )
            count += 1

        export_rows_by_demography_and_file(self.output_dir, rows, model_name)

        return count
    
def load_dimension_catalog(filepath: Path) -> DimensionCatalog:
    """Load stereotype dimension catalog from a Python module."""
    module_globals = runpy.run_path(str(filepath))
    data = module_globals.get("DIMENSION_CATALOG")
    if data is None:
        raise ValueError(
            f"Dimension catalog file must define DIMENSION_CATALOG: {filepath}",
        )
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
    demography_str, country = infer_group_metadata(filepath.stem)
    demography = Demography(demography_str)

    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            group = Group(
                group_id=row["group_id"].strip(),
                group_name=row["group_name"].strip(),
                demography=demography,
                country=country,
                source_file=filepath.stem,
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
