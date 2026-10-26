# IndoBias

Code for **IndoBias: A Dual Track Culturally Grounded Benchmark for LLMs Bias Evaluation in Indonesian Languages** ([arXiv:2606.01260](https://arxiv.org/abs/2606.01260)).

IndoBias evaluates social bias in language models in Indonesian and three local languages: Javanese, Sundanese and Makassar. It has two tracks:

| Track | Folder | What it measures |
|---|---|---|
| Depth (pair-based) | [`indobias_pair/`](indobias_pair) | Whether a model prefers the prototypical (stereotypical) sentence over its contrastive anti-stereotypical pair, scored by perplexity. Also covers the continued-pretraining experiments on Wikipedia, news and CC100. |
| Breadth (generation-based) | [`indobias_qa/`](indobias_qa) | Stereotype polarity in model responses to prompts about Indonesian groups (ethnicity, gender, religion, names, political parties, institutions, universities), grounded in SPI, O\*NET and WGI dimensions. |

> **Warning:** this benchmark deals with content that may be offensive, harmful or biased.

## Data access

The benchmark data and the raw model outputs are **not** included in this repository. They are available for research use on request. Please contact Ikhlasul Akmal Hanif (ikhlasulakmalh@gmail.com).

Once you have the data, place it at these paths (all git-ignored):

```
indobias_pair/data/IndoBias_{ID,Javanese,Sundanese,Makassar}.csv
indobias_qa/data/groups/*.csv
```

## Setup

```bash
uv sync
```

The pair-based track also has a pinned environment in `indobias_pair/requirements.txt`.

## Pair-based track

Choose the models and datasets in `indobias_pair/config.py`, then run:

```bash
python indobias_pair/calculate_ppl.py
```

For the continued-pretraining experiments, see `indobias_pair/run_scripts/` and [`indobias_pair/readme.md`](indobias_pair/readme.md). `post_pretraining_eval.py` scores the saved checkpoints. The scripts under `indobias_pair/analysis/` build the tables and figures used in the paper.

## Generation-based track

```bash
# Inspect tasks, dimensions and groups
python -m indobias_qa.commands list-tasks
python -m indobias_qa.commands stats

# Run a local Hugging Face model over all groups
python -m indobias_qa.model_runner --model-name Qwen/Qwen3-8B

# Run OpenAI models through the Batch API
python -m indobias_qa.commands openai-batch-prepare-submit
```

The `indobias_qa/run_allgroups_*.sh` scripts run the model sets used in the paper. `indobias_qa/analysis/` computes the stereotype-polarity summaries and spider charts, and `evaluate/analyze_lsi.py` computes the LLM Stereotype Index (LSI).

## Citation

```bibtex
@misc{hanif2026indobias,
  title         = {IndoBias: A Dual Track Culturally Grounded Benchmark for LLMs Bias Evaluation in Indonesian Languages},
  author        = {Hanif, Ikhlasul Akmal and Azmi, Muhammad Falensi and Tjiaranata, Filbert Aurelian and Yulianrifat, Eryawan Presma and Koto, Fajri},
  year          = {2026},
  eprint        = {2606.01260},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2606.01260}
}
```
