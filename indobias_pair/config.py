from dataclasses import dataclass
from itertools import product

# format the models into (type, model_name) (e.g., ('decoder','Qwen/Qwen3-8B'))
# Decoder families below are grouped by upstream foundation model lineage from
# the Hugging Face model cards, not by the publisher of the adapted model.
DECODER_MODELS: list[tuple[str, str]] = list(product(['decoder'], 
                                              [
                                               # Global base / instruct models
                                               # 'Qwen/Qwen3-8B-Base',
                                               # 'Qwen/Qwen3-8B',
                                               # 'meta-llama/Llama-3.1-8B',
                                               # 'meta-llama/Llama-3.1-8B-Instruct',
                                               # 'google/gemma-3-4b-pt',
                                               # 'google/gemma-3-4b-it',
                                               # 'allenai/Olmo-3-1025-7B',
                                               # 'allenai/Olmo-3-7B-Instruct',
                                               'meta-llama/Llama-2-7b-hf',
                                               'meta-llama/Llama-2-7b-chat-hf',
                                               'Qwen/Qwen2-7B',
                                               'Qwen/Qwen2-7B-Instruct',
                                               'Qwen/Qwen2.5-7B',
                                               'Qwen/Qwen2.5-7B-Instruct',
                                               'google/gemma-2-9b',
                                               'google/gemma-2-9b-it',
                                               'Open-Orca/Mistral-7B-OpenOrca',

                                               # SEA continued-pretraining / chat adaptation
                                               # HF lineage:
                                               # - aisingapore/Gemma-SEA-LION-v3-9B uses Gemma2
                                               # - SeaLLMs/SeaLLMs-v3-7B is a SEA-adapted Qwen2 base
                                               # - sail/Sailor2-8B builds on Qwen2.5
                                               # - Sahabat-AI/gemma2-9b-cpt-sahabatai-v1-* continues from SEA-LION v3
                                            #    'aisingapore/Gemma-SEA-LION-v3-9B',
                                            #    'aisingapore/Gemma-SEA-LION-v3-9B-IT',
                                            #    'SeaLLMs/SeaLLMs-v3-7B',
                                            #    'SeaLLMs/SeaLLMs-v3-7B-Chat',
                                            #    'sail/Sailor2-8B',
                                            #    'sail/Sailor2-8B-Chat',
                                            #    'Sahabat-AI/gemma2-9b-cpt-sahabatai-v1-base',
                                            #    'Sahabat-AI/gemma2-9b-cpt-sahabatai-v1-instruct',

                                               # Indonesian culturally tuned models
                                               # HF lineage:
                                               # - Yellow-AI-NLP/komodo-7b-base builds on Llama-2-7B-Base
                                               # - indonlp/cendol-llama2-7b-chat is the LLaMA-2 7B chat variant
                                               # - Ichsan2895/Merak-7B-v4 is based on Mistral-7B-OpenOrca
                                            #    'Yellow-AI-NLP/komodo-7b-base',
                                            #    'indonlp/cendol-llama2-7b-chat',
                                            #    'Ichsan2895/Merak-7B-v4',


                                               ]))
# Only include encoder checkpoints that expose a masked-language-model head.
# Embedding encoders such as `intfloat/multilingual-e5-base` cannot be scored
# through `lmppl.MaskedLM` pseudo-perplexity and will fail at load time.
ENCODER_MODELS: list[tuple[str, str]] = list(product(['encoder'], 
                                              [
                                                  # 'google-bert/bert-base-multilingual-cased',
                                               # 'FacebookAI/xlm-roberta-base',
                                               # 'indolem/indobert-base-uncased',
                                               # 'indolem/indobertweet-base-uncased',
                                               # "intfloat/multilingual-e5-large"
                                               
                                              ]))

# sample for smaller models
# DECODER_MODELS: list[str, str] = list(product(['decoder'], ['Qwen/Qwen3-0.6B-Base', 'Qwen/Qwen2.5-0.5B']))
# ENCODER_MODELS: list[str, str] = list(product(['encoder'], ['indolem/indobert-base-uncased', 'indolem/indobertweet-base-uncased']))

@dataclass
class ExperimentConfig:
    model_list: tuple[tuple[str, str], ...] = tuple(DECODER_MODELS + ENCODER_MODELS)
    file_to_process: tuple[str, ...] = (
        "indobias_pair/data/IndoBias_Javanese.csv",
        "indobias_pair/data/IndoBias_Makassar.csv",
        "indobias_pair/data/IndoBias_Sundanese.csv",
    )
    batch_size: int = 8
    device:str = 'cuda'
