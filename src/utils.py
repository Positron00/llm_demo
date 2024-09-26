import transformers
import random
import torch
import numpy as np

def model2hfname(modelName: str) -> str:
    return {}[modelName]

def dataset2hfname(dataset: str) -> str:
    return {}[dataset]

def get_dataset(dataset: str):
    pass

def get_model_and_tokenizer(modelName: str, modelClass, **model_kwargs):
    hf_model_name = model2hfname(modelName)

    model = modelClass.from_pretrained(hf_model_name, **model_kwargs)

    tokenizer = transformers.AutoTokenizer.from_pretrained(hf_model_name)

    if tokenizer.pad_token_id is None:
        if modelClass == transformers.AutoModelForCausalLM:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            print("Adding pad token to tokenizer")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tokenizer.pad_token = '[PAD]'
    return model, tokenizer


def fix_random_seeds(
        seed=123,
        set_system=True,
        set_torch=True):
    """
    Fix random seeds for reproducibility.

    Parameters:
    ----------
    seed : int
        Random seed to be set.
    set_system : bool
        Whether to set `np.random.seed(seed)` and `random.seed(seed)`
    set_torch : bool
        Whether to set `torch.manual_seed(seed)`
    """
    # set system seed
    if set_system:
        random.seed(seed)
        np.random.seed(seed)

    # set torch seed
    if set_torch:
        torch.manual_seed(seed)
