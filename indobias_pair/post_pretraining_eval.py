import os
import gc
import torch
import shutil
import argparse
import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download, hf_hub_download
from lmppl import MaskedLM

# ==========================================
# PARSE ARGS
# ==========================================
parser = argparse.ArgumentParser(description="Post-Pretraining Evaluation for Multiple Languages")
parser.add_argument("--repo_id", type=str, required=True, help="The Hugging Face repo ID (e.g., FrinzTheCoder/indobert-pt-cc100)")
args = parser.parse_args()

# ==========================================
# CONFIGURATION
# ==========================================
REPO_ID = args.repo_id
EXPERIMENT_NAME = REPO_ID.split("/")[-1]
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is not set. Please set it before running.")

EVAL_DATASET_NAME = "FrinzTheCoder/IndoBias-ADJ"
LANGUAGES = ["id", "javanese", "makassar", "sundanese"]
MAX_SEQ_LENGTH = 512
EVAL_BATCH_SIZE = 128

# Define the local root directory for saving results
LOCAL_RESULT_ROOT = os.path.join("result", "pretraining", "raw", EXPERIMENT_NAME)

api = HfApi(token=HF_TOKEN)

# ==========================================
# EVALUATION LOGIC
# ==========================================
def run_post_eval():
    print(f"🔍 Scanning Repository: {REPO_ID}")
    try:
        files = api.list_repo_files(repo_id=REPO_ID, repo_type="model")
    except Exception as e:
        print(f"❌ Could not access repository {REPO_ID}. Error: {e}")
        return

    # Extract all checkpoint folders (e.g., "checkpoint-25000")
    checkpoints = sorted(
        list(set(f.split('/')[0] for f in files if f.startswith("checkpoint-"))),
        key=lambda x: int(x.split('-')[1]) # Sort numerically by step
    )

    if not checkpoints:
        print("⚠️ No checkpoints found in this repository.")
        return

    print(f"✅ Found {len(checkpoints)} checkpoints. Starting evaluation loop...\n")

    for ckpt in checkpoints:
        print(f"==================================================")
        print(f"🚀 Processing {ckpt}...")
        
        # Ensure local result directory exists
        local_ckpt_result_dir = os.path.join(LOCAL_RESULT_ROOT, ckpt)
        os.makedirs(local_ckpt_result_dir, exist_ok=True)

        # 1. Check which languages are already evaluated in the Hub
        # We look for files named `bias_report_{lang}.csv` inside the checkpoint folder
        evaluated_langs = []
        for lang in LANGUAGES:
            expected_filename = f"{ckpt}/bias_report_{lang}.csv"
            if expected_filename in files:
                evaluated_langs.append(lang)

        missing_langs = [lang for lang in LANGUAGES if lang not in evaluated_langs]

        # 2. If all languages are evaluated, just download them locally and skip model loading
        if not missing_langs:
            print(f"    ✅ All {len(LANGUAGES)} languages already evaluated for {ckpt}.")
            for lang in LANGUAGES:
                local_file_path = os.path.join(local_ckpt_result_dir, f"bias_report_{lang}.csv")
                if not os.path.exists(local_file_path):
                    print(f"      -> Downloading existing report for [{lang}] to local directory...")
                    hf_hub_download(
                        repo_id=REPO_ID, 
                        filename=f"{ckpt}/bias_report_{lang}.csv", 
                        repo_type="model", 
                        local_dir=LOCAL_RESULT_ROOT, # hf_hub_download appends the filename path
                        token=HF_TOKEN
                    )
            continue

        # 3. If there are missing languages, we must download weights and run evaluations
        print(f"    ⚠️ Missing evaluations for: {missing_langs}")
        print(f"    -> Downloading weights for {ckpt}...")
        
        local_weights_dir = snapshot_download(
            repo_id=REPO_ID,
            repo_type="model",
            allow_patterns=f"{ckpt}/*",
            token=HF_TOKEN
        )
        
        model_path = os.path.join(local_weights_dir, ckpt)

        # Initialize the model exactly ONCE per checkpoint to save massive amounts of time/VRAM
        print(f"    -> Loading MaskedLM into VRAM...")
        scorer = MaskedLM(
            model=model_path, 
            max_length=MAX_SEQ_LENGTH,
            device_map="cuda"
        )

        # Evaluate all missing languages
        for lang in missing_langs:
            print(f"      -> 🧪 Evaluating subset [{lang}]...")
            try:
                # Load specific language subset
                ds = load_dataset(EVAL_DATASET_NAME, lang, split="train", token=HF_TOKEN)
                df = ds.to_pandas()
                df_clone = df.copy(deep=True) 
                
                protrope_texts = df_clone['protrope'].to_list()
                antitrope_texts = df_clone['antitrope'].to_list()

                inputs = []
                inputs.extend(protrope_texts)
                inputs.extend(antitrope_texts)
                
                # Get perplexities
                results = scorer.get_perplexity(inputs, batch_size=EVAL_BATCH_SIZE)
                
                # Calculate metrics
                df_clone['protrope_perplexity'] = results[:len(df_clone)]
                df_clone['antitrope_perplexity'] = results[len(df_clone):]
                df_clone['protrope_win'] = df_clone['protrope_perplexity'] < df_clone['antitrope_perplexity']
                
                # Save locally as CSV
                csv_filename = f"bias_report_{lang}.csv"
                local_csv_path = os.path.join(local_ckpt_result_dir, csv_filename)
                df_clone.to_csv(local_csv_path, index=False)
                print(f"         ✅ Saved locally: {local_csv_path}")

                # Upload to Hub
                print(f"         -> Uploading {csv_filename} to Hugging Face Hub...")
                api.upload_file(
                    path_or_fileobj=local_csv_path,
                    path_in_repo=f"{ckpt}/{csv_filename}",
                    repo_id=REPO_ID,
                    repo_type="model",
                    commit_message=f"Add evaluation for {lang} at {ckpt}"
                )
                print(f"         ✅ Successfully pushed {lang} report to Hub!")
                
            except Exception as e:
                print(f"         ❌ Failed evaluation for {lang}: {e}")

        # Clean up VRAM before moving to the next checkpoint
        print(f"    -> Cleaning up VRAM...")
        del scorer
        gc.collect()
        torch.cuda.empty_cache()

        # DELETE CHECKPOINT FROM DISK
        if os.path.exists(local_weights_dir):
            print(f"    -> 🗑️ Deleting checkpoint weights from {local_weights_dir} to save space...")
            shutil.rmtree(local_weights_dir)

    print("\n🎉 Post-Pretraining Evaluation Complete!")

if __name__ == "__main__":
    run_post_eval()