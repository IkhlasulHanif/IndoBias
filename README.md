<div align="center">
  <h1>IndoBias: A Dual Track Culturally Grounded Benchmark for LLMs Bias Evaluation in Indonesian Languages</h1>
</div>

<div align="center">
  <p style="font-size: 20px;">by
    <a href="https://scholar.google.com/citations?user=_l_utfMAAAAJ&hl=en">Ikhlasul Akmal Hanif</a><sup>1*</sup>,
    Muhammad Falensi Azmi<sup>1*</sup>,
    Filbert Aurelian Tjiaranata<sup>2</sup>,
    Eryawan Presma Yulianrifat<sup>2</sup>,
    <a href="https://www.fajrikoto.com/">Fajri Koto</a><sup>1</sup>
  </p>
  <p>
    <sup>*</sup>Main Authors<br>
    <sup>1</sup>MBZUAI, <sup>2</sup>Universitas Indonesia
  </p>

[![Project Page](https://img.shields.io/badge/Project-Page-0055be.svg)](https://ikhlasulhanif.github.io/IndoBias/)
[![arXiv](https://img.shields.io/badge/arXiv-2606.01260-b31b1b.svg)](https://arxiv.org/abs/2606.01260)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-yellow.svg)](https://huggingface.co/datasets/ikhlasulakmalh/IndoBias)
[![GitHub](https://img.shields.io/badge/GitHub-Code-blue?logo=github)](https://github.com/IkhlasulHanif/IndoBias)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

IndoBias evaluates social bias in language models in Indonesian and three local languages: Javanese, Sundanese and Makassar. It has two tracks:

| Track | Folder | What it measures |
|---|---|---|
| Depth (pair-based) | [`indobias_pair/`](indobias_pair) | Whether a model prefers the prototypical (stereotypical) sentence over its contrastive anti-stereotypical pair, scored by perplexity. Also covers the continued-pretraining experiments on Wikipedia, news and CC100. |
| Breadth (generation-based) | [`indobias_qa/`](indobias_qa) | Stereotype polarity in model responses to prompts about Indonesian groups (ethnicity, gender, religion, names, political parties, institutions, universities), grounded in SPI, O\*NET and WGI dimensions. |

> **Warning:** this benchmark deals with content that may be offensive, harmful or biased.

## Data access

The benchmark data is not included in this repository. It is hosted as a gated dataset on Hugging Face: [ikhlasulakmalh/IndoBias](https://huggingface.co/datasets/ikhlasulakmalh/IndoBias). Request access on that page, and the authors will review your request. For other questions, including access to the raw model outputs, contact Ikhlasul Akmal Hanif (ikhlasul.hanif@mbzuai.ac.ae) or Muhammad Falensi Azmi (muhammad.azmi@mbzuai.ac.ae).

Once approved, download the data into the paths the code expects (all git-ignored):

```bash
hf download ikhlasulakmalh/IndoBias --repo-type dataset --include "pair/*" --local-dir hf_data
mkdir -p indobias_pair/data indobias_qa/data/groups
cp hf_data/pair/*.csv indobias_pair/data/
hf download ikhlasulakmalh/IndoBias --repo-type dataset --include "qa/groups/*" --local-dir hf_data
cp hf_data/qa/groups/*.csv indobias_qa/data/groups/
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

## License

The code in this repository is released under the [MIT License](LICENSE). The benchmark data is not covered by this license; its terms are provided with the data on request.
