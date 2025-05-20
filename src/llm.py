from functools import lru_cache

import box
import yaml
from dotenv import find_dotenv, load_dotenv
from langchain.llms import LlamaCpp, CTransformers
from llama_cpp import Llama
from pydantic import PrivateAttr
from langchain.llms.base import LLM

N_CTX = 1024

from typing import Optional, List, Any

@lru_cache(maxsize=None)
def build_llm():
    load_dotenv(find_dotenv())
    with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
        cfg = box.Box(yaml.safe_load(ymlfile))
    print('loading ctransformer/llama')


    # raw_llm = Llama(
    #     model_path=cfg.MODEL_BIN_PATH,
    #     n_gpu_layers=cfg.GPU_LAYERS,
    #     n_ctx=N_CTX,       # Context size
    #     temperature=cfg.TEMPERATURE,
    #     n_threads=cfg.NUM_THREADS,
    #     seed=1337,
    #     verbose=True
    # )

    # print('transformer loaded successfully')

    # return raw_llm
    # # return CustomLlamaLLM(llm=raw_llm)

    kwargs = {
        "model_path": cfg.MODEL_BIN_PATH,
        "n_ctx": N_CTX,
        "max_tokens": cfg.MAX_NEW_TOKENS,
        "temperature": cfg.TEMPERATURE,
        "n_gpu_layers":cfg.GPU_LAYERS,
        "n_threads":cfg.NUM_THREADS,
        "n_batch": cfg.BATCH_SIZE,
        "verbose": True  # set this based on your GPU & CPU RAM
    }

    print('Running Mistral 7B')

    print('transformer loaded successfully')
    return LlamaCpp(**kwargs)


 
