import os
import random
from collections import defaultdict
import numpy as np
import json
from typing import List
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import transformers
from rouge_score import rouge_scorer
import tqdm

import utils

DEVICE = os.environ["DEVICE"] if "DEVICE" in os.environ else "cpu"

if DEVICE == "gpu" and torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device("mps")
elif DEVICE == "gpu" and torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("In-context learning using device: ", DEVICE)


def get_icl_prompts(
    support_inputs: List[str],
    support_labels: List[str],
    test_input: str,
    prompt_mode: str = 'qa') -> str:
    """
    Take a list of contexts and combine them into k-shot prompts.

    Args:
      support_inputs: The k inputs used for in-context learning (k may be zero)
      support_labels: The k labels used for in-context learning (k may be zero)
      test_input: The input we are evaluating on
      prompt_mode: 
        'none': only using k-shot examples;
        'tldr': using the TL;DR: prompt from the GPT-2 paper;
        'qa': adding "In the" after the question and before the answer;
        'custom': custom prompt format for article summarization. 
        
    Returns:
      A string containing the complete input to the model.
    """
    
    prompt = ''

    k = len(support_labels) # check how many support
    kOrder = np.random.permutation(k)

    # build the prompt
    if prompt_mode == 'qa':
        if k > 0:
            for i in kOrder:
                prompt = prompt + support_inputs[i] + ' In the ' + support_labels[i] + '. '
        prompt = prompt + test_input + ' In the' 

    elif prompt_mode == 'none':
        if k > 0:
            for i in kOrder:
                prompt = prompt + support_inputs[i] + ' ' + support_labels[i] + ' '
        prompt = prompt + test_input

    elif prompt_mode == 'tldr':
        if k > 0:
            for i in kOrder:
                prompt = prompt + support_inputs[i] + ' TL;DR: ' + support_labels[i] + '. '
        prompt = prompt + test_input + ' TL;DR:'

    elif prompt_mode == 'custom':
        if k > 0:
            for i in kOrder:
                prompt = prompt + support_inputs[i] + ' Summarized in 30 words as if explaining to a middle school student: ' + support_labels[i] + '. '
        prompt = prompt + test_input + ' Summarized in 30 words as if explaining to a middle school student:'

    return prompt


def get_performance_metric(predictions: List[str], targets: List[str], metric: str) -> float:
    """
    Calculate performance metric based on either rouge score or exact match.
    """
    if metric == 'rouge':
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        scores = []
        for p, t in zip(predictions, targets):
            score = scorer.score(p, t)['rouge1'].fmeasure
            scores.append(score)
        return sum(scores) / len(scores)
    elif metric == 'exact match':
        if isinstance(targets[0], str):
            return sum([p.strip() == t.strip() for p, t in zip(predictions, targets)]) / len(predictions)
        else:
            def _normalize(prediction):
                if prediction.endswith('Q'):
                    prediction = prediction[:-1]
                elif 'Q:' in prediction:
                    prediction = prediction[:prediction.index('Q:')]
                return prediction.strip('. ').lower()

            normalized = [_normalize(p) for p in predictions]
            def contains(key, candidates):
                for c in candidates:
                    if key in c:
                        return True
                return False

            return sum([contains(n, t) for n, t in zip(normalized, targets)]) / len(normalized)
    else:
        raise NotImplementedError()


def do_sample(model, input_ids, stop_tokens, max_tokens):
    """
    Sample from the model using the given input_ids as a prefix until either
    hitting the stop token or having sampled max_tokens tokens.

    Args:
        model: A transformers.PreTrainedModel to sample from
        input_ids: An integer tensor of shape [1, prefix_len]
        stop_tokens: A list of token ids for stopping sampling (e.g., a period)
        max_tokens: Stop sampling if already sampled this many tokens
    
    Returns:
        The sampled tokens (a python list of ints/zero-dim tensors), not including the input_ids prefix
          OR the stop token (if hitting the stop token before max_tokens)
    """

    sampled_tokens = []

    for _ in range(max_tokens):
        with torch.inference_mode():
            pred = model(input_ids)
        current_logits = pred.logits[:, -1, :]
        newToken = torch.argmax(current_logits,dim=-1)

        if newToken in stop_tokens:
            break
        else:
            sampled_tokens.append(newToken.item())
            input_ids = torch.cat((input_ids, newToken.unsqueeze(0)), 1)
         
    return sampled_tokens


def run_icl(models: List[str], datasets_: List[str], ks: List[int], prompt_modes: List[str], debug: bool, repeats: int, n_val: int = 125):
    results = {}

    for model_name in models:
        print(f'Loading model {model_name}...')

        utils.fix_random_seeds()

        model, tokenizer = utils.get_model_and_tokenizer(model_name, transformers.AutoModelForCausalLM)
                
        stop_tokens = utils.stop_tokens(tokenizer)
        model.to(DEVICE)

        for dataset in datasets_:
            print(f'Loading dataset {dataset}...')
            if debug:
                n_val = 1
            
            utils.fix_random_seeds()
            
            max_tokens = utils.max_sampled_tokens_for_dataset(dataset)
            train, val = utils.get_dataset(dataset, n_train=max(ks), n_val=n_val)

            for prompt_mode in prompt_modes:
                for k in ks:
                    print(f'Running in-context learning with {model_name} on {dataset} with k={k} and prompt_mode={prompt_mode}')

                    utils.fix_random_seeds()

                    for repeat in range(repeats):
                        if repeat > 0:
                            print(f'Beginning repeat #{repeat}')
                        support_idxs = random.choices(range(len(train['x'])), k=k)
                        support_x = [train['x'][idx].replace('\n', ' ') for idx in support_idxs]
                        support_y = [train['simple_y'][idx].replace('\n', ' ') for idx in support_idxs]
                        targets = []
                        predictions = []
                        pbar = tqdm.tqdm(list(range(min(n_val, len(val['x'])))))
                        for row in pbar:
                            test_input = val['x'][row]
                            targets.append(val['y'][row])
                            decoded_prediction = ''
                            prompt = get_icl_prompts(support_x,support_y,test_input,prompt_mode)
                            input_ids = tokenizer(prompt,return_tensors='pt').input_ids.to(DEVICE)
                            sampled_tokens = do_sample(model, input_ids, stop_tokens, max_tokens)
                            decoded_prediction = tokenizer.decode(sampled_tokens)

                            predictions.append(decoded_prediction)
                            metric = get_performance_metric(predictions, targets, utils.metric_for_dataset(dataset))
                            pbar.set_description(f'Eval: {metric:.04f}')

                        if metric >= results.get('_'.join([model_name, dataset, str(k), prompt_mode]), 0):
                            results['_'.join([model_name, dataset, str(k), prompt_mode])] = metric

                        print('Evaluation results:', results)
                        if not os.path.exists('results/icl'):
                            os.makedirs('results/icl')

                        for k_, v in results.items():
                            with open(f'results/icl/{k_}.json', 'w') as f:
                                json.dump({'metric': v}, f)
                        results = {}


def plot_icl(models, dataset, ks, prompt_modes, output):
    data = defaultdict(lambda: defaultdict(list))
    symbols = ['solid', 'dashed', 'dotted', 'dashdot']

    x_vals = set()
    for model in models:
        symbol = symbols.pop(0)
        for prompt_mode in prompt_modes:
            for k in ks:
                fn = '_'.join([model, dataset, str(k), prompt_mode])
                id_ = '_'.join([model, dataset, prompt_mode])
                with open(f'results/icl/{fn}.json', 'r') as f:
                    score = json.load(f)['metric']
                    data[id_]['x'].append(k)
                    x_vals.add(k)
                    data[id_]['y'].append(score)
                    data[id_]['linestyle'] = symbol

    for k, v in data.items():
        plt.plot(v['x'], v['y'], label=k, linestyle=v['linestyle'])

    if max(x_vals) > 4:
        plt.xscale('symlog')
    
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_ticks(v['x'])
    plt.legend()
    plt.title(dataset)
    plt.ylabel(utils.metric_for_dataset(dataset))
    plt.xlabel('Number of support examples')
    plt.savefig(output, bbox_inches='tight')
