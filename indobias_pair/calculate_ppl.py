import os
import gc
import shutil
import pandas as pd
import torch
import torch.nn.functional as F

from pathlib import Path

from utils.maskedlm_compat import patch_lmppl_transformers_compat
from lmppl import MaskedLM
from config import *
from dataclasses import fields
from argparse import ArgumentParser
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

patch_lmppl_transformers_compat()


REPO_ROOT = Path(__file__).resolve().parent
PAIR_RAW_RESULT_DIR = REPO_ROOT / "result" / "pairs" / "raw"


# function to load args from config.py
def add_args_from_dataclass(parser, dc, prefix=""):
    for f in fields(dc):
        arg_name = f"--{prefix}{f.name}"
        parser.add_argument(
            arg_name,
            default=f.default,
            type=f.type,
        )

# function to extract config
def extract_config(args, cls, prefix):
    kwargs = {
        f.name: getattr(args, f"{prefix}{f.name}")
        for f in fields(cls)
    }
    return cls(**kwargs)

# load config.py
parser = ArgumentParser()
add_args_from_dataclass(parser, ExperimentConfig, prefix="experiment.")
args = parser.parse_args()

# extract config
experiment_cfg = extract_config(args, ExperimentConfig, "experiment.")

files_to_process = experiment_cfg.file_to_process
model_list = experiment_cfg.model_list
batch_size = experiment_cfg.batch_size
device = experiment_cfg.device

def get_hf_home():
    return os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface"))


def delete_model_cache(model_id: str):
    repo_cache_dir = os.path.join(
        get_hf_home(),
        "hub",
        f"models--{model_id.replace('/', '--')}"
    )
    if os.path.exists(repo_cache_dir):
        shutil.rmtree(repo_cache_dir)
        print(f"🗑️ Deleted cached model from {repo_cache_dir}")


MASKED_LM_MODEL_TYPES = {
    "albert",
    "bert",
    "camembert",
    "deberta",
    "deberta-v2",
    "distilbert",
    "electra",
    "flaubert",
    "roberta",
    "xlm",
    "xlm-roberta",
}


def validate_encoder_model(model_id: str):
    try:
        config = AutoConfig.from_pretrained(model_id)
    except Exception as exc:
        raise ValueError(
            f"Could not load config for encoder model `{model_id}`"
        ) from exc

    architectures = config.architectures or []
    if any(arch.endswith("ForMaskedLM") for arch in architectures):
        return

    if getattr(config, "model_type", None) in MASKED_LM_MODEL_TYPES:
        return

    raise ValueError(
        f"Encoder model `{model_id}` is not a masked-LM checkpoint and "
        "cannot be evaluated with pseudo-perplexity."
    )


def _resolve_decoder_device(model):
    if hasattr(model, "device"):
        return model.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_decoder_scorer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map=device,
    )
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def get_decoder_perplexity(tokenizer, model, inputs: list[str], batch_size: int, max_length: int = 256):
    model_device = _resolve_decoder_device(model)
    perplexities: list[float] = []

    for batch_start in range(0, len(inputs), batch_size):
        batch = inputs[batch_start: batch_start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        encoded = {key: value.to(model_device) for key, value in encoded.items()}

        outputs = model(**encoded)
        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = encoded["input_ids"][:, 1:].contiguous()
        shift_attention = encoded["attention_mask"][:, 1:].contiguous()

        token_losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view(shift_labels.size())

        valid_token_count = shift_attention.sum(dim=1).clamp_min(1)
        sequence_nll = (token_losses * shift_attention).sum(dim=1) / valid_token_count
        perplexities.extend(torch.exp(sequence_nll).detach().cpu().tolist())

    return perplexities


dataset_frames = []
for file_to_process in files_to_process:
    dataset_frames.append(
        (
            os.path.splitext(os.path.basename(file_to_process))[0],
            pd.read_csv(file_to_process),
        )
    )


# evaluate models
for model_type, model_id in model_list:
    scorer = None
    decoder_tokenizer = None
    try:
        print(f"Evaluating model: {model_id}")
        if model_type == 'encoder':
            validate_encoder_model(model_id)
            scorer = MaskedLM(
                model=model_id,
                max_length=256,
                device_map=device,
            )
        elif model_type == 'decoder':
            decoder_tokenizer, scorer = build_decoder_scorer(model_id)

        for dataset_name, df in dataset_frames:
            print(f"  -> Running on {dataset_name}")

            df_clone = df.copy(deep=True) # deep clone df
            protrope_texts = df_clone['protrope'].to_list()
            antitrope_texts = df_clone['antitrope'].to_list()

            # reshape data into single list when calculating perplexity
            inputs = []
            inputs.extend(protrope_texts)
            inputs.extend(antitrope_texts)
            if model_type == 'decoder':
                results = get_decoder_perplexity(
                    tokenizer=decoder_tokenizer,
                    model=scorer,
                    inputs=inputs,
                    batch_size=batch_size,
                )
            else:
                results = scorer.get_perplexity(inputs, batch_size=batch_size)
            
            # reshape data back to original form
            df_clone['protrope_perplexity'] = results[:len(df_clone)]
            df_clone['antitrope_perplexity'] = results[len(df_clone):]

            # output comprehensive report
            output_path = PAIR_RAW_RESULT_DIR / dataset_name
            output_path.mkdir(parents=True, exist_ok=True)
            full_output_path = output_path / (
                f"{model_type}_{model_id.replace('/', '_').replace('.', '_')}.xlsx"
            )
            df_clone.to_excel(full_output_path)

    except Exception as e:
        print(f"❌ Failed for model {model_id} [{model_type}]. Skipping to next model: {e}")
        continue
    finally:
        if scorer is not None:
            del scorer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        delete_model_cache(model_id)
