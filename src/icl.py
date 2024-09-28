from typing import List
import torch
import numpy as np
import os
from rouge_score import rouge_scorer


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

    return prompt


def get_performance_metric(predictions: List[str], targets: List[str], metric: str) -> float:
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
