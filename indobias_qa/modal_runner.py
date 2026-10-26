"""Modal entry point for running IndoBias QA HF inference on GPUs."""

from __future__ import annotations

import io
import re
import tarfile
from pathlib import Path, PurePosixPath

import modal

APP_NAME = "indobias-qa"
VOLUME_NAME = "indobias-qa-results"
REMOTE_ROOT = PurePosixPath("/root/IndoBias")
REMOTE_OUTPUT_DIR = REMOTE_ROOT / "results" / "raw"
LOCAL_PACKAGE_DIR = Path(__file__).resolve().parent
LOCAL_SECRETS_DIR = LOCAL_PACKAGE_DIR / "models" / "secrets"
LOCAL_ROOT_SECRETS_DIR = LOCAL_PACKAGE_DIR.parent / "secrets"


def _ignore_package_path(path: Path) -> bool:
    path = Path(path)
    if path.name in {".DS_Store"}:
        return True
    if "__pycache__" in path.parts:
        return True

    try:
        relative = path.resolve().relative_to(LOCAL_PACKAGE_DIR)
    except ValueError:
        relative = Path(*path.parts[-3:])

    if relative.parts[:2] == ("models", "secrets"):
        return True
    if relative.parts[:1] == ("results",):
        return True
    if relative.parts[:1] == ("results 2",):
        return True
    return False


def _secret_env_name(secret_file: Path) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", secret_file.stem).strip("_").upper()
    aliases = {
        "HF": "HF_TOKEN",
        "HUGGINGFACE": "HF_TOKEN",
        "HUGGINGFACE_TOKEN": "HF_TOKEN",
        "OPENAI": "OPENAI_API_KEY",
        "OPENAI_API": "OPENAI_API_KEY",
        "OPENAI_API_KEY": "OPENAI_API_KEY",
    }
    return aliases.get(normalized, normalized)


def _load_local_secret_env() -> dict[str, str]:
    secret_env: dict[str, str] = {}
    for secrets_dir in (LOCAL_SECRETS_DIR, LOCAL_ROOT_SECRETS_DIR):
        if not secrets_dir.exists():
            continue
        for secret_file in sorted(secrets_dir.iterdir()):
            if not secret_file.is_file() or secret_file.name.startswith("."):
                continue
            value = secret_file.read_text(encoding="utf-8").strip()
            if not value:
                continue
            env_name = _secret_env_name(secret_file)
            secret_env[env_name] = value
            if env_name == "HF_TOKEN":
                secret_env.setdefault("HUGGING_FACE_HUB_TOKEN", value)
    return secret_env


def _archive_directory(directory: Path) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=path.relative_to(directory))
    return buffer.getvalue()


def _extract_archive(archive: bytes, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_root = output_dir.resolve()

    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for member in tar.getmembers():
            target = (output_root / member.name).resolve()
            if output_root not in (target, *target.parents):
                raise RuntimeError(f"Refusing to extract unsafe archive member: {member.name}")
        tar.extractall(output_root)


def _checkpoint_tag(
    model_name: str,
    tasks: str | None,
    demography: str | None,
    dimension: str | None,
    temperature: float,
    top_p: float,
    num_generations: int,
    sample_fraction: float | None,
    sample_seed: str,
) -> str:
    from indobias_qa.generator import normalize_model_name

    parts = [
        normalize_model_name(model_name),
        f"tasks-{tasks or 'all'}",
        f"demography-{demography or 'all'}",
        f"dimension-{dimension or 'all'}",
        f"temp-{temperature:g}",
        f"top_p-{top_p:g}",
        f"n-{num_generations}",
    ]
    if sample_fraction is not None:
        parts.append(f"sample-{sample_fraction:g}")
        parts.append(f"seed-{sample_seed}")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", "__".join(parts))


secret_env = _load_local_secret_env()
modal_secrets = [modal.Secret.from_dict(secret_env)]
results_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "accelerate>=0.30.0",
        "einops>=0.8.0",
        "openpyxl>=3.1.5",
        "pandas>=2.2.0",
        "protobuf>=5.0.0",
        "pydantic>=2.5.0",
        "rich>=13.7.0",
        "safetensors>=0.4.3",
        "sentencepiece>=0.2.0",
        "torch>=2.4.0",
        "tqdm>=4.66.0",
        "transformers>=4.53.0",
        "typer>=0.9.0",
    )
    .add_local_dir(
        str(LOCAL_PACKAGE_DIR),
        str(REMOTE_ROOT / "indobias_qa"),
        copy=True,
        ignore=_ignore_package_path,
    )
    .workdir(str(REMOTE_ROOT))
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    secrets=modal_secrets,
    timeout=6 * 60 * 60,
    volumes={str(REMOTE_OUTPUT_DIR): results_volume},
)
def run_remote(
    model_name: str,
    tasks: str | None,
    demography: str | None,
    dimension: str | None,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    num_generations: int,
    batch_size: int,
    enable_thinking: bool,
    sample_fraction: float | None,
    sample_seed: str,
    checkpoint_interval_batches: int,
    output_subdir: str | None,
) -> bytes:
    from pathlib import Path

    from indobias_qa.model_runner import run

    output_dir = Path(REMOTE_OUTPUT_DIR)
    if output_subdir:
        output_dir = output_dir / output_subdir
    checkpoint_path = output_dir / "_checkpoints" / (
        _checkpoint_tag(
            model_name,
            tasks,
            demography,
            dimension,
            temperature,
            top_p,
            num_generations,
            sample_fraction,
            sample_seed,
        )
        + ".jsonl"
    )
    run(
        dimensions_file=Path(REMOTE_ROOT / "indobias_qa" / "data" / "dimensions.py"),
        groups_path=Path(REMOTE_ROOT / "indobias_qa" / "data" / "groups"),
        output_dir=output_dir,
        model_name=model_name,
        tasks=tasks,
        demography=demography,
        dimension=dimension,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        num_generations=num_generations,
        batch_size=batch_size,
        device="cuda",
        enable_thinking=enable_thinking,
        sample_fraction=sample_fraction,
        sample_seed=sample_seed,
        checkpoint_path=checkpoint_path,
        checkpoint_interval_batches=checkpoint_interval_batches,
        checkpoint_callback=results_volume.commit,
    )
    results_volume.commit()
    return _archive_directory(output_dir)


@app.function(
    image=image,
    timeout=10 * 60,
    volumes={str(REMOTE_OUTPUT_DIR): results_volume},
)
def archive_remote_outputs() -> bytes:
    from pathlib import Path

    output_dir = Path(REMOTE_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    return _archive_directory(output_dir)


@app.local_entrypoint()
def main(
    model_name: str = "Qwen/Qwen3-8B",
    tasks: str | None = None,
    demography: str | None = None,
    dimension: str | None = None,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 0.9,
    num_generations: int = 1,
    batch_size: int = 1,
    gpu: str = "L40S",
    timeout_seconds: int = 6 * 60 * 60,
    output_dir: str = "results/raw",
    enable_thinking: bool = False,
    sample_fraction: float | None = None,
    sample_seed: str = "indobias-qa",
    checkpoint_interval_batches: int = 10,
    output_subdir: str | None = None,
) -> None:
    """Run IndoBias QA on Modal and extract result files locally."""
    archive = run_remote.with_options(gpu=gpu, timeout=timeout_seconds).remote(
        model_name=model_name,
        tasks=tasks,
        demography=demography,
        dimension=dimension,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        num_generations=num_generations,
        batch_size=batch_size,
        enable_thinking=enable_thinking,
        sample_fraction=sample_fraction,
        sample_seed=sample_seed,
        checkpoint_interval_batches=checkpoint_interval_batches,
        output_subdir=output_subdir,
    )
    _extract_archive(archive, Path(output_dir))
    print(f"Results extracted to {Path(output_dir).resolve()}")


@app.local_entrypoint()
def start(
    model_name: str = "Qwen/Qwen3-8B",
    tasks: str | None = None,
    demography: str | None = None,
    dimension: str | None = None,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 0.9,
    num_generations: int = 1,
    batch_size: int = 1,
    gpu: str = "L40S",
    timeout_seconds: int = 6 * 60 * 60,
    enable_thinking: bool = False,
    sample_fraction: float | None = None,
    sample_seed: str = "indobias-qa",
    checkpoint_interval_batches: int = 10,
    output_subdir: str | None = None,
) -> None:
    """Start an IndoBias QA Modal run without downloading outputs locally."""
    call = run_remote.with_options(gpu=gpu, timeout=timeout_seconds).spawn(
        model_name=model_name,
        tasks=tasks,
        demography=demography,
        dimension=dimension,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        num_generations=num_generations,
        batch_size=batch_size,
        enable_thinking=enable_thinking,
        sample_fraction=sample_fraction,
        sample_seed=sample_seed,
        checkpoint_interval_batches=checkpoint_interval_batches,
        output_subdir=output_subdir,
    )
    print(f"Started Modal function call: {call.object_id}")


@app.local_entrypoint()
def download(output_dir: str = "results/raw") -> None:
    """Download any files currently persisted in the Modal results volume."""
    archive = archive_remote_outputs.remote()
    _extract_archive(archive, Path(output_dir))
    print(f"Results extracted to {Path(output_dir).resolve()}")
