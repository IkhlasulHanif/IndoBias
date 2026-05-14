from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert JSONL files under analysis/ into XLSX files."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="analysis",
        help="JSONL file or directory to scan recursively. Defaults to analysis/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional base output directory. Relative layout is preserved under this directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .xlsx files.",
    )
    return parser.parse_args()


def find_jsonl_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix != ".jsonl":
            raise ValueError(f"Expected a .jsonl file, got: {input_path}")
        return [input_path]

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    return sorted(path for path in input_path.rglob("*.jsonl") if path.is_file())


def output_path_for(jsonl_path: Path, root: Path, output_dir: Path | None) -> Path:
    if output_dir is None:
        return jsonl_path.with_suffix(".xlsx")

    relative_path = jsonl_path.relative_to(root)
    return output_dir / relative_path.with_suffix(".xlsx")


def convert_jsonl_to_xlsx(jsonl_path: Path, xlsx_path: Path, overwrite: bool) -> None:
    if xlsx_path.exists() and not overwrite:
        print(f"Skipping existing file: {xlsx_path}")
        return

    df = pd.read_json(jsonl_path, lines=True)
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(xlsx_path, index=False)
    print(f"Wrote {xlsx_path}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    scan_root = input_path if input_path.is_dir() else input_path.parent
    jsonl_files = find_jsonl_files(input_path)

    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found under: {input_path}")

    for jsonl_path in jsonl_files:
        xlsx_path = output_path_for(jsonl_path, scan_root, args.output_dir)
        convert_jsonl_to_xlsx(jsonl_path, xlsx_path, args.overwrite)


if __name__ == "__main__":
    main()
