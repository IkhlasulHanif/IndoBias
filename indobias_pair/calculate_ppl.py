import os
import pandas as pd

from lmppl import MaskedLM
from evaluate import load
from config import *
from dataclasses import fields
from argparse import ArgumentParser

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

file_to_process = experiment_cfg.file_to_process
model_list = experiment_cfg.model_list
batch_size = experiment_cfg.batch_size
device = experiment_cfg.device

perplexity = load("perplexity", module_type="metric")

# data placeholder
data = dict()

# load
df = pd.read_csv(file_to_process)

# evaluate models
for model_type, model_id in model_list:
    try:
        print(f"Evaluating perplexity for model: {model_id}")

        df_clone = df.copy(deep=True) # deep clone df
        protrope_texts = df_clone['protrope'].to_list()
        antitrope_texts = df_clone['antitrope'].to_list()

        # reshape data into single list when calculating perplexity
        inputs = []
        inputs.extend(protrope_texts)
        inputs.extend(antitrope_texts)
        if model_type == 'decoder':
            results = perplexity.compute(
                predictions=inputs,
                model_id=model_id,
                batch_size=batch_size,
                device=device
            )
            results = results['perplexities']
        else:
            scorer = MaskedLM(model=model_id, 
                            max_length=256,
                            device_map=device)
            results = scorer.get_perplexity(inputs, batch_size=batch_size)
        
        # reshape data back to original form
        df_clone['protrope_perplexity'] = results[:len(df_clone)]
        df_clone['antitrope_perplexity'] = results[len(df_clone):]

        # output comprehensive report
        output_path = os.path.join('result','raw')
        os.makedirs(output_path, exist_ok=True)
        full_output_path = os.path.join(output_path, f"{model_type}_" + model_id.replace("/", "_").replace(".", "_")+".xlsx")
        df_clone.to_excel(full_output_path)

    except Exception as e:
        print(f"❌ Failed for model {model_id}: {e}")