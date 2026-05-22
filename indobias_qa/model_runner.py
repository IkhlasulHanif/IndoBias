"""Run local model inference over generated IndoBias prompts."""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from .config import DEFAULT_RAW_RESULTS_DIR, DIMENSIONS_FILE, GROUPS_DIR
from .generator import export_rows_by_demography_and_file, generate_prompts, load_dimension_catalog, load_groups_from_csv, load_groups_from_directory
from .models import HFChatModel
from .models.openai_batch import parse_demographies, parse_dimensions, parse_task_ids

def load_groups(groups_path: Path):
    if groups_path.is_dir():
        return load_groups_from_directory(groups_path)
    return load_groups_from_csv(groups_path)


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
    total = len(prompts)
    print(f"Total prompts to run: {total}")
    if total == 0:
        return

    model = HFChatModel(
        model_name=model_name,
        device=device,
        max_length=max_new_tokens,
        num_generations=num_generations,
        temperature=temperature,
        top_p=top_p,
        enable_thinking=enable_thinking,
    )

    rows: list[dict[str, object]] = []
    for i in tqdm(range(0, total, batch_size), desc=f"Running {model_name}"):
        prompt_batch = prompts[i:i + batch_size]
        responses_batch = model.batch_generate(
            [prompt.prompt_text for prompt in prompt_batch],
            num_generations=num_generations,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            batch_size=batch_size,
        )
        for prompt, responses in zip(prompt_batch, responses_batch):
            response = responses[0] if responses else ""
            metadata = prompt.metadata
            demography_value = metadata.demography.value
            source_file = metadata.source_file or demography_value
            rows.append(
                {
                    "model_name": model_name,
                    "prompt_id": metadata.prompt_id,
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
            )

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
    )


if __name__ == "__main__":
    main()
