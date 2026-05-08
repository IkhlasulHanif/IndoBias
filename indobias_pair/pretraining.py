import wandb
import weave
import os
import gc
import torch
import argparse
from lmppl import MaskedLM
from transformers import TrainerCallback
from huggingface_hub import HfApi, snapshot_download
from datasets import load_dataset, interleave_datasets
from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    AutoModelForMaskedLM, 
    DataCollatorForLanguageModeling, 
    TrainingArguments, 
    Trainer
)
from utils.dataset_cleaners import clean_liputan6_text
from utils.data_exporter import export_training_sample

# ==========================================
# PARSE ARGS
# ==========================================
parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, default="indolem/indobert-base-uncased")
parser.add_argument("--debug", action="store_true")

# Dataset 1 (Required)
parser.add_argument("--dataset1", type=str, required=True)
parser.add_argument("--variant1", type=str, default="default")
parser.add_argument("--prob1", type=float, default=None, help="Mixing weight for dataset 1") # ADDED

# Dataset 2 (Optional)
parser.add_argument("--dataset2", type=str, default=None)
parser.add_argument("--variant2", type=str, default="default")
parser.add_argument("--prob2", type=float, default=None, help="Mixing weight for dataset 2") # ADDED

# Dataset 3 (Optional)
parser.add_argument("--dataset3", type=str, default=None)
parser.add_argument("--variant3", type=str, default="default")
parser.add_argument("--prob3", type=float, default=None, help="Mixing weight for dataset 3") # ADDED

args = parser.parse_args()

# ==========================================
# CONFIG
# ==========================================

# NOTE: LOGIN TO WANDB
wandb.login()

# NOTE: LOAD HF TOKEN
HF_TOKEN = os.environ.get("HF_TOKEN")

DEBUG = args.debug

MODEL_NAME = args.model
MODEL_NAME_SHORT = MODEL_NAME.split("/")[-1]

# 1. Gather all inputted datasets dynamically
datasets_info = [(args.dataset1, args.variant1)]
dataset_probs = [args.prob1] if args.prob1 is not None else [] # ADDED

if args.dataset2:
    datasets_info.append((args.dataset2, args.variant2))
    if args.prob2 is not None:                                 # ADDED
        dataset_probs.append(args.prob2)                       # ADDED

if args.dataset3:
    datasets_info.append((args.dataset3, args.variant3))
    if args.prob3 is not None:                                 # ADDED
        dataset_probs.append(args.prob3)                       # ADDED

# ADDED: Normalize probabilities so they always sum to 1.0
if len(dataset_probs) > 0:
    if len(dataset_probs) != len(datasets_info):
        raise ValueError("If using mixing probabilities, you must provide a --prob flag for ALL inputted datasets.")
    total_prob = sum(dataset_probs)
    dataset_probs = [p / total_prob for p in dataset_probs]
else:
    dataset_probs = None


# 2. Create combined naming conventions for W&B and Hub
COMBINED_DATASET_NAME_SHORT = "_".join([d[0].split("/")[-1] for d in datasets_info])
COMBINED_VARIANT = "_".join([d[1] for d in datasets_info if d[1] != "default"])
if not COMBINED_VARIANT:
    COMBINED_VARIANT = "default"

WANDB_PROJECT = "indobias-pretraining-simulation"
if DEBUG:
    WANDB_RUN_NAME = f"2060-{MODEL_NAME_SHORT}-{COMBINED_DATASET_NAME_SHORT}-{COMBINED_VARIANT}-1"
else:
    WANDB_RUN_NAME = f"GPUXX-{MODEL_NAME_SHORT}-{COMBINED_DATASET_NAME_SHORT}-{COMBINED_VARIANT}-1"

# training setup
HUB_MODEL_ID = f"FrinzTheCoder/{MODEL_NAME_SHORT}-pt-{COMBINED_DATASET_NAME_SHORT}-{COMBINED_VARIANT}"
EVAL_DATASET_NAME = "FrinzTheCoder/IndoBias-ADJ"
EVAL_BATCH_SIZE = 128
if DEBUG:
    MAX_SEQ_LENGTH = 64
    training_args = TrainingArguments(
        output_dir=f"./indobias-pretraining-simulation-{MODEL_NAME_SHORT}-{COMBINED_DATASET_NAME_SHORT}-{COMBINED_VARIANT}",
        per_device_train_batch_size=8,
        max_steps=200,
        gradient_accumulation_steps=1,
        warmup_steps=10,
        save_steps=100,
        logging_steps=10,
        num_train_epochs=1,
        save_total_limit=1,
        save_strategy="steps",
        learning_rate=1e-4,
        weight_decay=0.01,
        seed=3407,
        report_to="wandb",
        push_to_hub=False,
        bf16=False,
        fp16=False,
    )
else:
    MAX_SEQ_LENGTH = 512
    training_args = TrainingArguments(
        output_dir=f"./indobias-pretraining-simulation-{MODEL_NAME_SHORT}-{COMBINED_DATASET_NAME_SHORT}-{COMBINED_VARIANT}",
        per_device_train_batch_size=64,
        max_steps=500_000,
        gradient_accumulation_steps=1,
        save_steps=25_000,
        warmup_steps=10_000,
        logging_steps=1_000,
        num_train_epochs=1,
        save_total_limit=1,
        save_strategy="steps",
        learning_rate=1e-4,
        weight_decay=0.01,
        seed=3407,
        report_to="wandb",
        push_to_hub=False,
        bf16=True,
        fp16=False,
        # --- A100 / H100 SPECIFIC OPTIMIZATIONS ---
        tf32=True,                  # Enables massive speedups on Ampere/Hopper GPUs
        dataloader_num_workers=4,   # Prevents CPU bottlenecking during streaming
        torch_compile=True,         # Compiles the model for ~20% faster training (PyTorch 2.0+)
    )

# ==========================================
# CODE TO EVALAUTE PSEUDO-PERPLEXITY
# ==========================================
def run_pseudo_perplexity_eval(checkpoint_dir, step, train_dataset_name, eval_dataset_name, hf_token):
    train_dataset_id = train_dataset_name.split('/')[-1].lower()
    print(f"   -> 🧪 Evaluating against [{eval_dataset_name}] for model trained on [{train_dataset_id}] (Step {step})...")
    
    try:
        ds = load_dataset(eval_dataset_name, token=hf_token)
        df = ds["train"].to_pandas()
        df_clone = df.copy(deep=True) 
        
        protrope_texts = df_clone['protrope'].to_list()
        antitrope_texts = df_clone['antitrope'].to_list()

        inputs = []
        inputs.extend(protrope_texts)
        inputs.extend(antitrope_texts)
        
        print(f"      -> Loading local checkpoint weights from: {checkpoint_dir}")
        scorer = MaskedLM(
            model=checkpoint_dir, 
            max_length=MAX_SEQ_LENGTH,
            device_map="cuda"
        )
        results = scorer.get_perplexity(inputs, batch_size=EVAL_BATCH_SIZE)
        
        # 1. Assign results directly
        df_clone['protrope_perplexity'] = results[:len(df_clone)]
        df_clone['antitrope_perplexity'] = results[len(df_clone):]

        # 2. Calculate Win Rate (Protrope wins if its perplexity is LOWER)
        df_clone['protrope_win'] = df_clone['protrope_perplexity'] < df_clone['antitrope_perplexity']

        # 3. Group by Domain to get the Win Rate (mean of boolean True/False = percentage)
        domain_win_rates = df_clone.groupby('domain')['protrope_win'].mean().to_dict()
        
        # Calculate overall mean across all domains for a high-level metric
        domain_win_rates['Overall'] = df_clone['protrope_win'].mean()

        # 4. Save to Excel
        excel_filename = f"bias_report_train_{train_dataset_id}_step_{step}.xlsx"
        full_output_path = os.path.join(checkpoint_dir, excel_filename)
        df_clone.to_excel(full_output_path, index=False)
        print(f"      ✅ Report successfully generated: {excel_filename}")

        # Memory cleanup
        del scorer
        gc.collect()
        torch.cuda.empty_cache()

        # Return the clean dictionary of scores for W&B logging
        return domain_win_rates

    except Exception as e:
        print(f"      ❌ Failed evaluation for step {step}: {e}")
        return None

# ==========================================
# CODE FOR AUTOMATED REPORTING
# ==========================================
class CustomEvalAndUploadCallback(TrainerCallback):
    def __init__(self, repo_id, token, train_dataset_name, eval_dataset_name, train_dataset, tokenizer):
        self.api = HfApi(token=token)
        self.repo_id = repo_id
        self.train_dataset_name = train_dataset_name
        self.eval_dataset_name = eval_dataset_name
        self.token = token
        
        # Save dataset and tokenizer for data exporting
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer
        self.has_exported_sample = False
        
        self.api.create_repo(
            repo_id=self.repo_id, 
            repo_type="model", 
            exist_ok=True, 
            private=True
        )

    def on_save(self, args, state, control, **kwargs):
        checkpoint_folder = f"checkpoint-{state.global_step}"
        local_checkpoint_path = os.path.join(args.output_dir, checkpoint_folder)
        
        print(f"\n🚀 Intercepting Save for {checkpoint_folder}...")
        
        # 1. Export training data sample (Only runs on the FIRST save step)
        if not self.has_exported_sample:
            export_training_sample(
                dataset=self.train_dataset,
                tokenizer=self.tokenizer,
                output_dir=local_checkpoint_path,
                step=state.global_step,
                num_samples=50
            )
            self.has_exported_sample = True

        # 2. Capture the dictionary of domain scores
        domain_scores = run_pseudo_perplexity_eval(
            checkpoint_dir=local_checkpoint_path, 
            step=state.global_step,
            train_dataset_name=self.train_dataset_name,
            eval_dataset_name=self.eval_dataset_name,
            hf_token=self.token
        )
        
        # 3. Push to W&B
        if domain_scores is not None and wandb.run is not None:
            print("   -> 📊 Pushing domain bias scores to W&B...")
            # Format metrics with a prefix so they group neatly in the UI
            wandb_metrics = {f"bias_win_rate/{domain}": score for domain, score in domain_scores.items()}
            # Create an explicit, un-shiftable X-axis variable
            wandb_metrics["checkpoint_step"] = state.global_step
            # Log all domains at the exact current global step
            wandb.log(wandb_metrics, step=state.global_step)
            
        # 4. Upload to Hub
        print(f"   -> Uploading weights and evaluation report to Hub...")
        try:
            self.api.upload_folder(
                folder_path=local_checkpoint_path,
                path_in_repo=checkpoint_folder,
                repo_id=self.repo_id,
                repo_type="model",
                commit_message=f"Upload {checkpoint_folder} + Bias Eval + Data Sample"
            )
            print(f"✅ Successfully backed up {checkpoint_folder} to the Hub!")
        except Exception as e:
            print(f"❌ Upload failed: {e}")

# ==========================================
# CODE FOR DOWNLOADING LATEST CHECKPOINT
# ==========================================
def download_latest_checkpoint(repo_id, local_dir, token):
    """Queries the HF Hub for the latest checkpoint and downloads only that folder."""
    api = HfApi(token=token)
    try:
        print(f"🔍 Checking Hugging Face Hub ({repo_id}) for previous checkpoints...")
        files = api.list_repo_files(repo_id=repo_id, repo_type="model")
        
        # Extract unique checkpoint folders (e.g., "checkpoint-200")
        checkpoint_folders = set()
        for f in files:
            if f.startswith("checkpoint-"):
                checkpoint_folders.add(f.split('/')[0])
                
        if not checkpoint_folders:
            print("   No checkpoints found. Starting from step 0.")
            return False
            
        # Find the folder with the highest step number
        latest_folder = max(checkpoint_folders, key=lambda x: int(x.split('-')[1]))
        print(f"   Found {latest_folder}! Downloading to {local_dir}...")
        
        # Download strictly this folder using allow_patterns
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=local_dir,
            allow_patterns=f"{latest_folder}/*",
            token=token
        )
        print("✅ Download complete.")
        return True
        
    except Exception as e:
        print(f"   ⚠️ Hub check failed (likely first run or repo doesn't exist yet). Details: {e}")
        return False

# ==========================================
# CODE FOR LOADING SINGLE DATASET
# ==========================================
def load_single_dataset(name, variant):
    """Loads a single dataset and enforces a strict 'text-only' schema for interleaving."""
    print(f"   -> Loading stream for {name} ({variant})...")
    if name == "SEACrowd/cc100":
        dataset = load_dataset(name, variant, split="train", streaming=True, trust_remote_code=True)
    elif name == "wikimedia/wikipedia":
        dataset = load_dataset(name, variant, split="train", streaming=True)
    elif name == "SEACrowd/liputan6":
        dataset = load_dataset(name, split="train", streaming=True)
        print(f"      -> Applying Liputan6 regex cleaner...")
        dataset = dataset.map(clean_liputan6_text, batched=True)
    elif name in ["SEACrowd/indo4b", "SEACrowd/indo4b_plus"]:
        raise NotImplementedError("TODO: IMPLEMENT")
    else:
        dataset = load_dataset(name, split='train', streaming=True)

    # FIX: Remove all columns except 'text' so datasets can interleave safely
    sample_row = next(iter(dataset))
    columns_to_remove = [col for col in sample_row.keys() if col != "text"]
    dataset = dataset.remove_columns(columns_to_remove)

    # Shuffle the individual stream
    return dataset.shuffle(buffer_size=10_000, seed=3407)

def setup_and_train():
    with wandb.init(
        project=WANDB_PROJECT,
        id=WANDB_RUN_NAME,
        resume="allow"
    ) as run:

        print("Loading Tokenizer and Datasets...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # 1. Load all specified datasets
        loaded_streams = [load_single_dataset(name, var) for name, var in datasets_info]

        # 2. Interleave if there are multiple datasets
        if len(loaded_streams) > 1:
            print(f"🔄 Interleaving {len(loaded_streams)} datasets together...")
            if dataset_probs:                                               # ADDED
                print(f"   -> Mixing proportions: {dataset_probs}")         # ADDED
            
            # ADDED probabilities parameter
            dataset = interleave_datasets(loaded_streams, probabilities=dataset_probs)
        else:
            dataset = loaded_streams[0]

        def tokenize_function(examples):
            return tokenizer(examples["text"], return_special_tokens_mask=True, truncation=False)

        # No need to dynamically find columns_to_remove here, we already standardized to just "text"
        tokenized_datasets = dataset.map(
            tokenize_function, 
            batched=True, 
            remove_columns=["text"]
        )

        # 2. The standard chunking function
        def group_texts(examples):
            # Concatenate all texts in the batch
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            # Compute total length
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            
            # We drop the small remainder at the very end of the batch
            # (Alternatively, you could add padding here, but dropping a few tokens out of millions is standard)
            if total_length >= MAX_SEQ_LENGTH:
                total_length = (total_length // MAX_SEQ_LENGTH) * MAX_SEQ_LENGTH
                
            # Split by chunks of max_len
            result = {
                k: [t[i : i + MAX_SEQ_LENGTH] for i in range(0, total_length, MAX_SEQ_LENGTH)]
                for k, t in concatenated_examples.items()
            }
            return result

        # 3. Apply the chunking to the tokenized dataset
        chunked_datasets = tokenized_datasets.map(group_texts, batched=True)

        # 4. Pass the chunked_datasets to the Trainer
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.15
        )

        # Initialize Model FROM SCRATCH
        print("Initializing architecture from scratch...")
        config = AutoConfig.from_pretrained(MODEL_NAME)
        # This loads the architecture but uses randomized weights, dropping pretrained knowledge
        model = AutoModelForMaskedLM.from_config(config)

        # Pad vocab size to multiple of 8
        vocab_size = len(tokenizer)
        if vocab_size % 8 != 0:
            padded_vocab_size = (vocab_size + 7) // 8 * 8
            print(f"Padding vocabulary from {vocab_size} to {padded_vocab_size} to align Tensor Cores...")
            model.resize_token_embeddings(padded_vocab_size)

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=chunked_datasets,
            data_collator=data_collator,
            callbacks=[
                CustomEvalAndUploadCallback(
                    repo_id=HUB_MODEL_ID, 
                    token=HF_TOKEN,
                    train_dataset_name=COMBINED_DATASET_NAME_SHORT,
                    eval_dataset_name=EVAL_DATASET_NAME,
                    train_dataset=chunked_datasets,  # Pass dataset to callback
                    tokenizer=tokenizer              # Pass tokenizer to callback
                )
            ]
        )

        # ==========================================
        # THE RESUME LOGIC
        # ==========================================
        LOCAL_CHECKPOINT_DIR = training_args.output_dir
        
        # 1. Try to pull the latest checkpoint from the Hub
        has_hub_checkpoint = download_latest_checkpoint(
            repo_id=HUB_MODEL_ID, 
            local_dir=LOCAL_CHECKPOINT_DIR, 
            token=HF_TOKEN
        )

        # 2. Tell the Trainer to resume if we found one
        if has_hub_checkpoint:
            print("🚀 Resuming training from downloaded checkpoint...")
            trainer.train(resume_from_checkpoint=True)
        else:
            print("🚀 Starting fresh training run...")
            trainer.train()

        # Save the final model
        trainer.save_model(os.path.join(LOCAL_CHECKPOINT_DIR, "final-model"))

if __name__ == "__main__":
    setup_and_train()