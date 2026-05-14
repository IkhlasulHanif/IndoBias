from dataclasses import dataclass
from itertools import product

# format the models into (type, model_name) (e.g., ('decoder','Qwen/Qwen3-8B'))
DECODER_MODELS: list[str, str] = list(product(['decoder'], 
                                              ['Qwen/Qwen3-8B-Base',
                                               'Qwen/Qwen3-8B',
                                               'meta-llama/Llama-3.1-8B',
                                               'meta-llama/Llama-3.1-8B-Instruct',
                                               'google/gemma-3-4b-pt',
                                               'google/gemma-3-4b-it',
                                               'allenai/Olmo-3-1025-7B',
                                               'allenai/Olmo-3-7B-Instruct',
                                               'aisingapore/Gemma-SEA-LION-v3-9B',
                                               'aisingapore/Gemma-SEA-LION-v3-9B-IT',
                                               'SeaLLMs/SeaLLMs-v3-7B',
                                               'SeaLLMs/SeaLLMs-v3-7B-Chat',
                                               'sail/Sailor2-8B',
                                               'sail/Sailor2-8B-Chat',
                                               'Sahabat-AI/gemma2-9b-cpt-sahabatai-v1-base',
                                               'Sahabat-AI/gemma2-9b-cpt-sahabatai-v1-instruct',
                                               'Yellow-AI-NLP/komodo-7b-base',
                                               'indonlp/cendol-llama2-7b-chat',
                                               'Ichsan2895/Merak-7B-v4',
                                               ]))
ENCODER_MODELS: list[str, str] = list(product(['encoder'], 
                                              ['google-bert/bert-base-multilingual-cased',
                                               'FacebookAI/xlm-roberta-base',
                                               'intfloat/multilingual-e5-base',
                                               'indolem/indobert-base-uncased',
                                               'indolem/indobertweet-base-uncased',
                                              ]))

# sample for smaller models
# DECODER_MODELS: list[str, str] = list(product(['decoder'], ['Qwen/Qwen3-0.6B-Base', 'Qwen/Qwen2.5-0.5B']))
# ENCODER_MODELS: list[str, str] = list(product(['encoder'], ['indolem/indobert-base-uncased', 'indolem/indobertweet-base-uncased']))

@dataclass
class ExperimentConfig:
    model_list: tuple[str, str] = tuple(DECODER_MODELS + ENCODER_MODELS)
    file_to_process: tuple[str, ...] = (
        "indobias_pair/data/IndoBias_Javanese.csv",
        "indobias_pair/data/IndoBias_Makassar.csv",
        "indobias_pair/data/IndoBias_Sundanese.csv",
    )
    batch_size: int = 8
    device:str = 'cuda'
