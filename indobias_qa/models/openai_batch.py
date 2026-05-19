from __future__ import annotations

import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import DEFAULT_RAW_RESULTS_DIR, DIMENSIONS_FILE, GROUPS_DIR
from ..data_models import Demography, GroupCollection, StereotypeDimension
from ..generator import (
    count_prompts,
    generate_prompts,
    load_dimension_catalog,
    load_groups_from_csv,
    load_groups_from_directory,
    normalize_model_name,
)

OPENAI_API_BASE = "https://api.openai.com/v1"
DEFAULT_MODELS = ("gpt-4.1-mini", "gpt-5-mini")
DEFAULT_BATCH_ROOT = Path("results") / "batch" / "openai"
TERMINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled"}


@dataclass
class BatchRecord:
    model_name: str
    request_file: str
    metadata_file: str
    input_file_id: str | None = None
    batch_id: str | None = None
    status: str | None = None
    output_file_id: str | None = None
    error_file_id: str | None = None
    output_file: str | None = None
    errors_file: str | None = None


def parse_task_ids(value: str | None) -> list[int] | None:
    if value is None or value.lower() == "all":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_demographies(value: str | None) -> list[Demography] | None:
    if value is None:
        return None
    return [Demography(value.lower())]


def parse_dimensions(value: str | None) -> list[StereotypeDimension] | None:
    if value is None:
        return None
    return [StereotypeDimension(value.lower())]


def load_groups(groups_path: Path) -> GroupCollection:
    if groups_path.is_dir():
        return load_groups_from_directory(groups_path)
    return load_groups_from_csv(groups_path)


def load_inputs(
    dimensions_file: Path,
    groups_path: Path,
    tasks: str | None,
    demography: str | None,
    dimension: str | None,
) -> tuple[Any, GroupCollection, list[int] | None, list[Demography] | None, list[StereotypeDimension] | None]:
    return (
        load_dimension_catalog(dimensions_file),
        load_groups(groups_path),
        parse_task_ids(tasks),
        parse_demographies(demography),
        parse_dimensions(dimension),
    )


def iter_batch_rows(output_file: Path, metadata_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with output_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            result = json.loads(line)
            metadata = metadata_by_id.get(result["custom_id"])
            if metadata is None:
                continue

            response_obj = result.get("response") or {}
            body = response_obj.get("body") or {}
            usage = body.get("usage") or {}
            error = result.get("error")
            rows.append(
                {
                    "prompt_id": metadata["prompt_id"],
                    "demography": metadata["demography"],
                    "source_file": metadata.get("source_file") or metadata["demography"],
                    "group_id": metadata["group_id"],
                    "group": metadata["group"],
                    "dimension": metadata["dimension"],
                    "dimension_name": metadata.get("dimension_name"),
                    "stereotype_pair": metadata["stereotype_pair"],
                    "parent_category": metadata.get("parent_category"),
                    "positive_label": metadata["positive_label"],
                    "negative_label": metadata["negative_label"],
                    "task_id": metadata["task_id"],
                    "task_name": metadata["task_name"],
                    "task_complexity_level": metadata["task_complexity_level"],
                    "prompt": metadata["prompt"],
                    "response": extract_response_text(body),
                    "response_status_code": response_obj.get("status_code"),
                    "request_id": response_obj.get("request_id"),
                    "error": json.dumps(error, ensure_ascii=False) if error else None,
                    "input_tokens": usage.get("input_tokens"),
                    "output_tokens": usage.get("output_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                }
            )
    return rows


def output_key(model_name: str, row: dict[str, Any]) -> tuple[str, str, str]:
    return (row["demography"], normalize_model_name(model_name), row["source_file"])


def prepare_batches(
    *,
    batch_root: Path = DEFAULT_BATCH_ROOT,
    dimensions_file: Path = DIMENSIONS_FILE,
    groups_path: Path = GROUPS_DIR,
    tasks: str | None = None,
    demography: str | None = None,
    dimension: str | None = None,
    models: list[str] | tuple[str, ...] = DEFAULT_MODELS,
    max_output_tokens: int = 1024,
) -> None:
    requests_dir = batch_root / "requests"
    requests_dir.mkdir(parents=True, exist_ok=True)
    (batch_root / "outputs").mkdir(parents=True, exist_ok=True)

    catalog, groups, task_ids, demographies, dimensions = load_inputs(
        dimensions_file,
        groups_path,
        tasks,
        demography,
        dimension,
    )

    total = count_prompts(groups, catalog, task_ids, demographies, dimensions)
    print(f"Preparing {total} prompts across {len(models)} model(s).")

    prompts = list(generate_prompts(groups, catalog, task_ids, demographies, dimensions))
    manifest: dict[str, dict[str, Any]] = load_manifest(batch_root)

    for model_name in models:
        normalized_model = normalize_model_name(model_name)
        request_path = requests_dir / f"{normalized_model}.requests.jsonl"
        metadata_path = requests_dir / f"{normalized_model}.metadata.jsonl"

        with request_path.open("w", encoding="utf-8") as request_file, metadata_path.open("w", encoding="utf-8") as metadata_file:
            for prompt in prompts:
                custom_id = prompt.metadata.prompt_id
                request = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": build_request_body(model_name, prompt.prompt_text, max_output_tokens),
                }
                request_file.write(json.dumps(request, ensure_ascii=False) + "\n")

                metadata = prompt.metadata.model_dump()
                metadata["prompt"] = prompt.prompt_text
                metadata["custom_id"] = custom_id
                metadata_file.write(json.dumps(metadata, ensure_ascii=False) + "\n")

        manifest[model_name] = asdict(
            BatchRecord(
                model_name=model_name,
                request_file=str(request_path),
                metadata_file=str(metadata_path),
            )
        )
        print(f"Wrote {request_path}")
        print(f"Wrote {metadata_path}")

    save_manifest(batch_root, manifest)


def build_request_body(model_name: str, prompt_text: str, max_output_tokens: int) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model_name,
        "input": prompt_text,
        "max_output_tokens": max_output_tokens,
    }
    if model_name.startswith("gpt-5"):
        body["reasoning"] = {"effort": "none"}
        body["text"] = {"verbosity": "low"}
    return body


def submit_batches(batch_root: Path = DEFAULT_BATCH_ROOT) -> None:
    manifest = load_manifest(batch_root)
    api_key = require_api_key()

    for model_name, record in manifest.items():
        if record.get("batch_id"):
            print(f"Skipping {model_name}: batch already exists ({record['batch_id']}).")
            continue

        request_file = Path(record["request_file"])
        input_file_response = upload_batch_file(request_file, api_key)
        batch_response = create_batch(input_file_response["id"], api_key)

        record["input_file_id"] = input_file_response["id"]
        record["batch_id"] = batch_response["id"]
        record["status"] = batch_response["status"]
        record["output_file_id"] = batch_response.get("output_file_id")
        record["error_file_id"] = batch_response.get("error_file_id")
        print(f"Submitted {model_name}: batch_id={record['batch_id']}")

    save_manifest(batch_root, manifest)


def refresh_status(batch_root: Path = DEFAULT_BATCH_ROOT) -> dict[str, dict[str, Any]]:
    manifest = load_manifest(batch_root)
    api_key = require_api_key()
    for model_name, record in manifest.items():
        batch_id = record.get("batch_id")
        if not batch_id:
            continue
        response = api_request(f"/batches/{batch_id}", api_key, method="GET")
        record["status"] = response["status"]
        record["output_file_id"] = response.get("output_file_id")
        record["error_file_id"] = response.get("error_file_id")
        print(f"{model_name}: {record['status']}")
    save_manifest(batch_root, manifest)
    return manifest


def watch_status(batch_root: Path = DEFAULT_BATCH_ROOT, interval_seconds: int = 30) -> None:
    while True:
        manifest = refresh_status(batch_root)
        statuses = {record.get("status") for record in manifest.values() if record.get("batch_id")}
        if statuses and statuses.issubset(TERMINAL_BATCH_STATUSES):
            return
        time.sleep(interval_seconds)


def download_outputs(batch_root: Path = DEFAULT_BATCH_ROOT) -> None:
    manifest = load_manifest(batch_root)
    api_key = require_api_key()
    outputs_dir = batch_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    for model_name, record in manifest.items():
        output_file_id = record.get("output_file_id")
        if record.get("status") != "completed" or not output_file_id:
            print(f"Skipping {model_name}: batch not completed.")
            continue

        normalized_model = normalize_model_name(model_name)
        output_path = outputs_dir / f"{normalized_model}.output.jsonl"
        errors_path = outputs_dir / f"{normalized_model}.errors.jsonl"

        download_file_content(output_file_id, output_path, api_key)
        record["output_file"] = str(output_path)
        print(f"Downloaded {output_path}")

        error_file_id = record.get("error_file_id")
        if error_file_id:
            download_file_content(error_file_id, errors_path, api_key)
            record["errors_file"] = str(errors_path)
            print(f"Downloaded {errors_path}")

    save_manifest(batch_root, manifest)


def export_outputs(
    batch_root: Path = DEFAULT_BATCH_ROOT,
    raw_output_dir: Path = DEFAULT_RAW_RESULTS_DIR,
) -> None:
    manifest = load_manifest(batch_root)
    grouped_rows: dict[tuple[str, str, str], list[dict[str, Any]]] = {}

    for model_name, record in manifest.items():
        output_file = record.get("output_file")
        metadata_file = record.get("metadata_file")
        if not output_file or not metadata_file:
            print(f"Skipping {model_name}: missing downloaded output or metadata file.")
            continue

        metadata_by_id = load_jsonl_index(Path(metadata_file), "custom_id")

        for row in iter_batch_rows(Path(output_file), metadata_by_id):
            row["model_name"] = model_name
            grouped_rows.setdefault(output_key(model_name, row), []).append(row)

    for (demography, normalized_model, source_file), rows in grouped_rows.items():
        target_dir = raw_output_dir / demography
        target_dir.mkdir(parents=True, exist_ok=True)
        output_path = target_dir / f"{normalized_model}__{source_file}.xlsx"
        pd.DataFrame(rows).to_excel(output_path, index=False)
        print(f"Wrote {output_path}")


def extract_response_text(body: dict[str, Any]) -> str | None:
    output_text = body.get("output_text")
    if isinstance(output_text, str):
        return output_text

    output_items = body.get("output") or []
    collected: list[str] = []
    for item in output_items:
        content_items = item.get("content") or []
        for content in content_items:
            text = content.get("text")
            if isinstance(text, str):
                collected.append(text)
    if collected:
        return "\n".join(collected).strip()
    return None


def load_jsonl_index(path: Path, key_field: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            result[row[key_field]] = row
    return result


def upload_batch_file(path: Path, api_key: str) -> dict[str, Any]:
    with path.open("rb") as handle:
        content = handle.read()

    boundary = f"----indobias-{uuid.uuid4().hex}"
    content_type = mimetypes.guess_type(path.name)[0] or "application/jsonl"
    body = build_multipart_body(
        boundary,
        {
            "purpose": ("text/plain", b"batch"),
            "file": (content_type, content, path.name),
        },
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    }
    return api_request("/files", api_key, method="POST", data=body, headers=headers, decode_json=True)


def create_batch(input_file_id: str, api_key: str) -> dict[str, Any]:
    payload = {
        "input_file_id": input_file_id,
        "endpoint": "/v1/responses",
        "completion_window": "24h",
    }
    return api_request("/batches", api_key, method="POST", json_body=payload)


def download_file_content(file_id: str, destination: Path, api_key: str) -> None:
    content = api_request(f"/files/{file_id}/content", api_key, method="GET", decode_json=False)
    destination.write_bytes(content)


def api_request(
    path: str,
    api_key: str,
    *,
    method: str,
    json_body: dict[str, Any] | None = None,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    decode_json: bool = True,
) -> Any:
    request_headers = {"Authorization": f"Bearer {api_key}"}
    payload = data
    if json_body is not None:
        payload = json.dumps(json_body).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    if headers:
        request_headers.update(headers)

    request = urllib.request.Request(
        f"{OPENAI_API_BASE}{path}",
        data=payload,
        headers=request_headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request) as response:
            content = response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(exc.read().decode("utf-8")) from exc

    if not decode_json:
        return content
    return json.loads(content.decode("utf-8"))


def build_multipart_body(boundary: str, fields: dict[str, tuple[Any, ...]]) -> bytes:
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        if name == "file":
            content_type, content, filename = value
            chunks.append(
                (
                    f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                    f"Content-Type: {content_type}\r\n\r\n"
                ).encode("utf-8")
            )
            chunks.append(content)
            chunks.append(b"\r\n")
        else:
            _, content = value
            chunks.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
            chunks.append(content)
            chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks)


def require_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required.")
    return api_key


def load_manifest(batch_root: Path = DEFAULT_BATCH_ROOT) -> dict[str, dict[str, Any]]:
    manifest_path = batch_root / "batches.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def save_manifest(batch_root: Path, manifest: dict[str, dict[str, Any]]) -> None:
    batch_root.mkdir(parents=True, exist_ok=True)
    manifest_path = batch_root / "batches.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
