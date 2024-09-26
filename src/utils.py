from collections import defaultdict
import datasets
import transformers
import logging
import random
import torch
import numpy as np
import os

def model2hfname(modelID: str) -> str:
    return {
        'gpt2-med': 'gpt2-medium',
        'Llama-3.1-8B': 'meta-llama/Meta-Llama-3.1-8B',
        'Llama-3.2-11B-V-Inst': 'meta-llama/Llama-3.2-11B-Vision-Instruct'
    }[modelID]

def dataset2hfname(datasetID: str) -> str:
    return {
        'amazon': ('amazon_us_reviews', 'Video_v1_00'),
        'xsum': ('xsum',),
        'babi': ('babi_qa', 'en-valid-10k-qa1')
    }[datasetID]

def get_dataset(datasetID: str, n_train: int, n_val: int = 100):
    if datasetID == 'babi':
        n_train = 256
        d = datasets.load_dataset('babi_qa', 'en-valid-10k-qa1', split='train')
        answer_idxs = []
        for story in d['story']:
            for idx, answer in enumerate(story['answer']):
                if answer:
                    answer_idxs.append(idx)
                    break
        
        perm = np.random.permutation(len(d['story']))
        answers = [story['answer'][idx] for idx, story in zip(answer_idxs, d['story'])]
        stories = [' '.join(story['text'][:idx + 1]) for idx, story in zip(answer_idxs, d['story'])]

        answers = [answers[idx] for idx in perm]
        stories = [stories[idx] for idx in perm]
        data = {'x': stories, 'y': answers, 'simple_y': answers}
        d = datasets.Dataset.from_dict(data)
        return d[:n_train], d[n_train:n_train + n_val]
    
    elif datasetID == 'amazon':
        # d = datasets.load_dataset('amazon_us_reviews', 'Video_v1_00')['train']
        data_files = "data/amazon_reviews_us_Video_v1_00.csv"
        try:
            d = datasets.load_dataset("csv", data_files=data_files)["train"]
        except FileNotFoundError:
            print(
                "PLEASE DOWNLOAD THE AMAZON DATASET FROM https://drive.google.com/file/d/1UMj_oWyGH4xorNkXyPrqIHQJn4CydDUd/view?usp=sharing AND PLACE IT IN data/amazon_reviews_us_Video_v1_00.csv"
            )
            exit(1)
        filter_fn = lambda rows: [r is None or 'sex' not in r.lower() for r in rows['review_body']]
        d = d.filter(filter_fn, batched=True, batch_size=None)
        x = d['review_body']
        y = [s - 1 for s in d['star_rating']]
        train = defaultdict(lambda: [None] * 5 * n_train)
        val = defaultdict(lambda: [None] * 5 * n_val)
        counts = defaultdict(int)
        for idx in range(len(y)):
            c = counts[y[idx]]
            if c < n_train:
                train['x'][c * 5 + y[idx]] = x[idx]
                train['y'][c * 5 + y[idx]] = y[idx]
                counts[y[idx]] += 1
            elif c < n_train + n_val:
                val['x'][(c - n_train) * 5 + y[idx]] = x[idx]
                val['y'][(c - n_train) * 5 + y[idx]] = y[idx]
                counts[y[idx]] += 1
        return train, val
    
    elif datasetID == 'xsum':
        n_train = 256
        d = datasets.load_dataset('xsum', split='train')
        filter_fn = lambda rows: [len(a.split(' ')) + len(s.split(' ')) < 100 for a, s in zip(rows['document'], rows['summary'])]
        d = d.filter(filter_fn, batched=True, batch_size=None)
        d = d.rename_columns({'document': 'x', 'summary': 'y'})
        d = d.add_column('simple_y', d['y'])
        return d[:n_train], d[n_train:n_train + n_val]
    
    else:
        raise NotImplementedError(f'{datasetID}')

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
