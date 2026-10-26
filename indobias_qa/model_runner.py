"""Run local model inference over generated IndoBias prompts."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Callable

from tqdm import tqdm

from .config import DEFAULT_RAW_RESULTS_DIR, DIMENSIONS_FILE, GROUPS_DIR
from .generator import export_rows_by_demography_and_file, generate_prompts, load_dimension_catalog, load_groups_from_csv, load_groups_from_directory
from .models import HFChatModel
from .models.openai_batch import parse_demographies, parse_dimensions, parse_task_ids

def load_groups(groups_path: Path):
    if groups_path.is_dir():
        return load_groups_from_directory(groups_path)
    return load_groups_from_csv(groups_path)


def sample_prompts_by_task(prompts, sample_fraction: float | None, sample_seed: str):
    if sample_fraction is None:
        return prompts
    if not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction must be greater than 0 and at most 1")

    sampled = []
    task_ids = sorted({prompt.metadata.task_id for prompt in prompts})
    for task_id in task_ids:
        task_prompts = [prompt for prompt in prompts if prompt.metadata.task_id == task_id]
        sample_size = max(1, round(len(task_prompts) * sample_fraction))
        ranked = sorted(
            task_prompts,
            key=lambda prompt: hashlib.sha256(
                f"{sample_seed}:{prompt.metadata.prompt_id}".encode("utf-8")
            ).hexdigest(),
        )
        sampled.extend(ranked[:sample_size])

    return sorted(sampled, key=lambda prompt: prompt.metadata.prompt_id)


def run(
    *,
    dimensions_file: Path,
    groups_path: Path,
    output_dir: Path,
    model_name: str,
    tasks: str | None,
    demography: str | None,
    dimension: str | None,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    num_generations: int,
    batch_size: int,
    device: str | None,
    enable_thinking: bool,
    sample_fraction: float | None = None,
    sample_seed: str = "indobias-qa",
    checkpoint_path: Path | None = None,
    checkpoint_interval_batches: int = 10,
    checkpoint_callback: Callable[[], None] | None = None,
) -> None:
    catalog = load_dimension_catalog(dimensions_file)
    groups = load_groups(groups_path)
    prompts = list(
        generate_prompts(
            groups,
            catalog,
            parse_task_ids(tasks),
            parse_demographies(demography),
            parse_dimensions(dimension),
        )
    )
    full_total = len(prompts)
    prompts = sample_prompts_by_task(prompts, sample_fraction, sample_seed)
    total = len(prompts)
    if sample_fraction is not None:
        print(
            f"Sampled {total} prompts from {full_total} "
            f"({sample_fraction:.2%}, seed={sample_seed})"
        )
    print(f"Total prompts to run: {total}")
    if total == 0:
        return

    rows: list[dict[str, object]] = []
    completed_generations_by_prompt: dict[str, set[int]] = defaultdict(set)
    if checkpoint_path is not None and checkpoint_path.exists():
        with checkpoint_path.open("r", encoding="utf-8") as checkpoint_file:
            for line in checkpoint_file:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows.append(row)
                prompt_id = str(row["prompt_id"])
                generation_index = int(row.get("generation_index", 0))
                completed_generations_by_prompt[prompt_id].add(generation_index)
        completed_generations = sum(
            len(generation_indices)
            for generation_indices in completed_generations_by_prompt.values()
        )
        if completed_generations:
            print(
                f"Resuming from {checkpoint_path}: "
                f"{completed_generations} completed generations"
            )

    model = HFChatModel(
        model_name=model_name,
        device=device,
        max_length=max_new_tokens,
        num_generations=num_generations,
        temperature=temperature,
        top_p=top_p,
        enable_thinking=enable_thinking,
    )

    pending_prompts = [
        prompt
        for prompt in prompts
        if len(completed_generations_by_prompt[prompt.metadata.prompt_id])
        < num_generations
    ]
    if not pending_prompts:
        export_rows_by_demography_and_file(output_dir, rows, model_name)
        print(f"Wrote {len(rows)} model responses to {output_dir}")
        return

    checkpoint_handle = None
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_handle = checkpoint_path.open("a", encoding="utf-8")

    batches_completed = 0
    for i in tqdm(range(0, len(pending_prompts), batch_size), desc=f"Running {model_name}"):
        prompt_batch = pending_prompts[i:i + batch_size]
        responses_batch = model.batch_generate(
            [prompt.prompt_text for prompt in prompt_batch],
            num_generations=num_generations,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            batch_size=batch_size,
        )
        for prompt, responses in zip(prompt_batch, responses_batch):
            metadata = prompt.metadata
            demography_value = metadata.demography.value
            source_file = metadata.source_file or demography_value
            completed_generation_indices = completed_generations_by_prompt[
                metadata.prompt_id
            ]
            for generation_index, response in enumerate(responses):
                if generation_index in completed_generation_indices:
                    continue
                row = {
                    "model_name": model_name,
                    "prompt_id": metadata.prompt_id,
                    "generation_index": generation_index,
                    "demography": demography_value,
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
                    "response": response,
                }
                rows.append(row)
                completed_generation_indices.add(generation_index)
                if checkpoint_handle is not None:
                    checkpoint_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        batches_completed += 1
        if checkpoint_handle is not None:
            checkpoint_handle.flush()
        if (
            checkpoint_callback is not None
            and checkpoint_interval_batches > 0
            and batches_completed % checkpoint_interval_batches == 0
        ):
            checkpoint_callback()

    if checkpoint_handle is not None:
        checkpoint_handle.close()
        if checkpoint_callback is not None:
            checkpoint_callback()

    export_rows_by_demography_and_file(output_dir, rows, model_name)
    print(f"Wrote {len(rows)} model responses to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local model inference for IndoBias prompts.")
    parser.add_argument("--dimensions", type=Path, default=DIMENSIONS_FILE)
    parser.add_argument("--groups", "-g", type=Path, default=GROUPS_DIR)
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_RAW_RESULTS_DIR)
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--tasks", "-t")
    parser.add_argument("--demography", "-d")
    parser.add_argument("--dimension", "-D")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--num-generations", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--device")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--sample-fraction", type=float)
    parser.add_argument("--sample-seed", default="indobias-qa")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(
        dimensions_file=args.dimensions,
        groups_path=args.groups,
        output_dir=args.output,
        model_name=args.model_name,
        tasks=args.tasks,
        demography=args.demography,
        dimension=args.dimension,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        num_generations=args.num_generations,
        batch_size=args.batch_size,
        device=args.device,
        enable_thinking=args.enable_thinking,
        sample_fraction=args.sample_fraction,
        sample_seed=args.sample_seed,
    )


if __name__ == "__main__":
    main()
