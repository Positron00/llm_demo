from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers

import utils
import copy
import numpy as np
import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import itertools
import tqdm

# determine device to train on
DEVICE = os.environ["DEVICE"] if "DEVICE" in os.environ else "cpu"

if DEVICE == "gpu" and torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device("mps")
elif DEVICE == "gpu" and torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("Fine-tuning using device: ", DEVICE)


def parameters_to_fine_tune(model: nn.Module, mode: str) -> List:
    """
    Select the parameters in `model` that should be fine-tuned in mode `mode`.

    Args:
      model: the model we're fine-tuning
      mode: the fine-tuning mode we're using; may be 'all', 'last', 'first',
        'middle'
    
    Returns:
      A list of nn.Parameters of `model` that should be fine-tuned in the given
        fine-tuning mode.
    """

    if mode == 'all':
        return model.parameters()
    elif mode == 'last':
        for param in model.parameters():
            param.requires_grad = False

        for param in model.transformer.h[-2:].parameters():
            param.requires_grad = True

        return filter(lambda p: p.requires_grad, model.parameters())
        
    elif mode == 'first':
        for param in model.parameters():
            param.requires_grad = False

        for param in model.transformer.h[:2].parameters():
            param.requires_grad = True

        return filter(lambda p: p.requires_grad, model.parameters())
    elif mode == 'middle':
        for param in model.parameters():
            param.requires_grad = False

        transformer_blocks = model.transformer.h
        mid_index = len(transformer_blocks) // 2
        parameters = list(transformer_blocks[mid_index - 1].parameters()) + list(transformer_blocks[mid_index].parameters())

        return parameters
    
    else:
        raise NotImplementedError()


def get_loss(logits: torch.tensor, targets: torch.tensor) -> torch.tensor:
    """
    Computes the cross-entropy loss.

    Args:
      logits: a 2D [batch_size, n_classes] (for classification) 
      targets: a 1D [batch_size] (for classification) tensor of target indices. 

    Returns:
      A zero-dim tensor representing the average cross-entropy loss over all batch 
        elements (and sequence timesteps, if applicable)
    """

    return F.cross_entropy(logits, targets).mean()

def get_acc(logits, targets):
    """
    Computes the exact match accuracy 

    Args:
      logits: a 2D [batch_size, n_classes] (for classification) tensor of logits
      targets: a 1D [batch_size] (for classification) tensor of target indices. 
    
    Returns:
      A *scalar* representing the average exact-match accuracy over all non-masked batch 
        elements
    """

    y = torch.argmax(logits, dim=-1) == targets
    y = y.type(torch.float)
    return torch.mean(y).item()
    

def ft_llm(model, tokenizer, x, y, mode, debug, batch_size=8):
    model = copy.deepcopy(model)

    optimizer = torch.optim.Adam(parameters_to_fine_tune(model, mode), lr=1e-4)
    all_x = tokenizer(x, return_tensors='pt', padding=True, truncation=True, max_length=100).to(DEVICE)
    all_y = torch.tensor(y, device=DEVICE)
    pbar = tqdm.tqdm(range(1000))
    for step in pbar:
        batch = np.random.randint(0, len(x), batch_size)
        x_ = tokenizer([x[i] for i in batch], return_tensors='pt', padding=True, truncation=True, max_length=100).to(DEVICE)
        y_ = torch.tensor([y[i] for i in batch], device=DEVICE)
        logits = model(**x_).logits
        loss = get_loss(logits, y_)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if debug:
            break

        if step % 10 == 0:
            with torch.inference_mode():
                total_acc = get_acc(model(**all_x).logits, all_y)
            pbar.set_description(f'Fine-tuning acc: {total_acc:.04f}')
            if total_acc > 0.75:
                break
    return model

def run_ft(models: List[str], datasets: List[str], ks: List[int], modes: List[str], debug: bool, repeats: int, n_val: int = 125):
    results = {}
    for dataset in datasets:
        utils.fix_random_seeds()

        if debug:
            n_val = 1   
        train, val = utils.get_dataset(dataset, max(ks), n_val=n_val)

        for model_name, mode in itertools.product(models, modes):
            utils.fix_random_seeds()
            model, tokenizer = utils.get_model_and_tokenizer(model_name, transformers.AutoModelForCausalLM)

            for k in ks:
                print(f'Fine-tuning {model_name} on {dataset} with k={k} and mode={mode}')
                utils.fix_random_seeds()

                for repeat in range(repeats):
                    if repeat > 0:
                        print(f'Beginning repeat #{repeat}')

                    fine_tuned = ft_llm(model, tokenizer, train['x'][:k], train['simple_y'][:k], mode, dataset)

                    fine_tuned.eval()
                  
                    print(results)
                    results = {}


def plot_ft(models, datasets, ks, modes, output):
    data = defaultdict(lambda: defaultdict(list))
    question = 'ft'

    x_vals = set()
    for dataset in datasets:
        for model, mode in itertools.product(models, modes):
            for k in ks:
                fn = '_'.join([model, dataset, str(k), mode])
                id_ = '_'.join([model, dataset, mode])
                with open(f'results/{question}/{fn}.json', 'r') as f:
                    score = json.load(f)['metric']
                    data[id_]['x'].append(k)
                    x_vals.add(k)
                    data[id_]['y'].append(score)

        for k, v in data.items():
            plt.plot(v['x'], v['y'], label=k)

    if max(x_vals) > 4:
        plt.xscale('symlog')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_ticks(sorted(x_vals))
    plt.legend()
    plt.title(' & '.join(datasets))
    plt.ylabel('/'.join([utils.metric_for_dataset(dataset) for dataset in datasets]))
    plt.xlabel('Number of support examples')
    plt.savefig(output, bbox_inches='tight')