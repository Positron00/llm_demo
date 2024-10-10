from typing import List, Tuple

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
from icl import get_icl_prompts, do_sample, get_performance_metric
import tqdm
import random


# determine device to train on
DEVICE = os.environ["DEVICE"] if "DEVICE" in os.environ else "cpu"

if DEVICE == "gpu" and torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device("mps")
elif DEVICE == "gpu" and torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("Fine-tuning using device: ", DEVICE)


class LoRAConv1DWrapper(nn.Module):
    # set up LoRA-augmented layer
    def __init__(self, conv1dmodule: nn.Module, lora_rank: int):
        super().__init__()

        self.base_module = conv1dmodule
        
        for param in self.base_module.parameters():
            param.requires_grad = False

        self.lora_A = nn.Parameter(torch.empty(self.base_module.weight.shape[0], lora_rank))
        nn.init.kaiming_uniform_(self.lora_A)
        self.lora_B = nn.Parameter(torch.zeros(self.base_module.weight.shape[1], lora_rank))
        self.lora_A.requires_grad = True
        self.lora_B.requires_grad = True

    def forward(self, x):
        base_output = self.base_module(x)
        lora_output = torch.matmul(torch.matmul(x,self.lora_A),self.lora_B.T)
        return base_output + lora_output


def parameters_to_fine_tune(model: nn.Module, mode: str) -> List:
    """
    Select the parameters in `model` that should be fine-tuned in mode `mode`.

    Args:
      model: the model we're fine-tuning
      mode: the fine-tuning mode we're using; may be 'all', 'last', 'first',
        'middle', or 'loraN' (where N is an integer)
    
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

        #for param in model.transformer.h[(mid_index-1):(mid_index+1)].parameters():
        #    param.requires_grad = True
        #parameters = filter(lambda p: p.requires_grad, model.parameters())
        
        parameters = list(transformer_blocks[mid_index - 1].parameters()) + list(transformer_blocks[mid_index].parameters())

        for param in parameters:
            param.requires_grad = True

        # Verification
        print(f"Number of parameters with gradients enabled: {sum(p.requires_grad for p in model.parameters())}")
        print(f"Total number of parameters: {len(parameters)}")

        return filter(lambda p: p.requires_grad, model.parameters())

    elif mode.startswith('lora'):
        for param in model.parameters():
            param.requires_grad = False

        for m in model.modules():
            if isinstance(m, LoRAConv1DWrapper):
                for name, param in m.named_parameters():
                    if 'lora_A' in name or 'lora_B' in name:
                        param.requires_grad = True

        return filter(lambda p: p.requires_grad, model.parameters())

    else:
        raise NotImplementedError()


def get_loss(logits: torch.tensor, targets: torch.tensor) -> torch.tensor:
    """
    Computes the cross-entropy loss for either sequence classification or generation.

    For generation, note different sequences within the batch have different lengths, 
      and the targets tensor includes some mask values (-100). The average loss is the 
      *average loss over all non-masked timesteps*. The prediction for what token t will 
      be is made after seeing only t - 1 tokens; that is, there is an off-by-one shift 
      needed between the logits and targets.

    Args:
      logits: a 2D [batch_size, n_classes] (for classification) or 3D
        [batch_size, sequence_length, vocab_size] (for generation) tensor
        of *UNNORMALIZED* logits
      targets: a 1D [batch_size] (for classification) or 2D [batch_size, sequence_length]
        (for generation) tensor of target indices. For the generation case, may contain
        -100 in some positions, meaning that the loss for this timestep should be ignored.
    
    Returns:
      A zero-dim tensor representing the average cross-entropy loss over all batch 
        elements (and sequence timesteps, if applicable)
    """

    if logits.dim() == 2:
        loss = F.cross_entropy(logits, targets).mean()

    elif logits.dim() == 3:
        logits = logits[:, :-1, :].contiguous()
        targets = targets[:, 1:].contiguous()

        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)

        loss = F.cross_entropy(logits, targets, ignore_index=-100)

    else:
        raise ValueError(f'Logits should either be 2-dim (for classification) or 3-dim (for generation); got {logits.dim()}')

    print(f"Loss requires_grad: {loss.requires_grad}")
    print(f"Loss has grad_fn: {loss.grad_fn is not None}")
    
    return loss


def get_acc(logits, targets):
    """
    Computes the exact match accuracy for either sequence classification or generation. i.e.,
      the fraction of predictions for which the most likely class/token equals the target.

    For generation, need to deal with the fact that different sequences within
      the batch have different lengths, and the targets tensor includes some mask
      values (-100). The average accuracy is the *average accuracy over all non-masked batch 
        elements (and sequence timesteps, if applicable)
      Also need to handle the fact that the prediction for what token t will be is
      made after seeing only t - 1 tokens; that is, there is an off-by-one shift needed
      between the logits and targets.

    Args:
      logits: a 2D [batch_size, n_classes] (for classification) or 3D
        [batch_size, sequence_length, vocab_size] (for generation) tensor of logits
      targets: a 1D [batch_size] (for classification) or 2D [batch_size, sequence_length]
        (for generation) tensor of target indices. For the generation case, may contain
        -100 in some positions, meaning that the loss for this timestep should be ignored.
    
    Returns:
      A *scalar* representing the average exact-match accuracy over all non-masked batch 
        elements (and sequence timesteps, if applicable)
    """

    if logits.dim() == 2:
        y = torch.argmax(logits, dim=-1) == targets
        y = y.type(torch.float)
        return torch.mean(y).item()
    
    elif logits.dim() == 3:
        logits = logits[:, :-1, :].contiguous()
        targets = targets[:, 1:].contiguous()

        y = torch.argmax(logits, dim=-1)
        mask = (targets != -100).float()
        correct = (y == targets).float() * mask

        return (correct.sum() / mask.sum()).item()
    
    else:
        raise ValueError(f'Logits should either be 2-dim (for classification) or 3-dim (for generation); got {logits.dim()}')


def ft_bert(model, tokenizer, x, y, mode, debug, batch_size=8):
    model = copy.deepcopy(model)

    if mode.startswith('lora'):
        for m in model.transformer.h:
            m.mlp.c_fc = LoRAConv1DWrapper(m.mlp.c_fc, int(mode[4:]))
            m.mlp.c_proj = LoRAConv1DWrapper(m.mlp.c_proj, int(mode[4:]))

    model.to(DEVICE)

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
            pbar.set_description(f'Fine-tuning training accuracy: {total_acc:.04f}')
            if total_acc > 0.75:
                break

    return model


def tokenize_gpt2_batch(tokenizer, x, y):
    """
    Implement the tokenization step for a batch of examples for GPT-2.

    Args:
        tokenizer: a GPT2Tokenizer that one can call and receive a dictionary of:
          - input_ids: a list (or tensor) of token ids
          - attention_mask: a list (or tensor) of 1s and 0s indicating which tokens
              are padding (if you requested padding and tensors from the tokenizer)
        x: a list of strings, each of which is the input for a single example
        y: a list of strings, each of which is a *target* for a single example
    
    Returns:
        A dictionary with the following keys:
            - input_ids: a tensor of shape [batch_size, sequence_length] 
                containing the token ids
            - attention_mask: a tensor of shape [batch_size, sequence_length] 
                containing 1s and 0s indicating which tokens are padding
            - labels: a tensor of shape [batch_size, sequence_length] containing
                the target token ids, with -100 for non-target tokens (i.e., the
                tokens in the input part of each example or padding tokens)
        where sequence_length is determined by the (x, y) pair whose tokenized
        length is the longest in the batch. The other sequences should be padded to
        this length (the tokenizer can handle this padding).
    """

    combined_sequences = [x_ + y_ for x_, y_ in zip(x, y)]
    tokenized_sequences = tokenizer(combined_sequences,return_tensors="pt",padding=True,truncation=True)

    labels = torch.full_like(tokenized_sequences['input_ids'], -100)
    tokenized_targets = tokenizer(y,return_tensors="pt",padding=True,truncation=True)
    tokenized_inputs = tokenizer(x,return_tensors="pt",padding=True,truncation=True)

    for i in range(len(labels)):
        inputTokens = tokenized_inputs['input_ids'][i]
        targetTokens = tokenized_targets['input_ids'][i]
        inputMask = tokenized_inputs['attention_mask'][i]
        targetMask = tokenized_targets['attention_mask'][i]

        lenInput = sum(inputMask)
        lenTarget = sum(targetMask)

        for j in range(lenInput,(lenInput+lenTarget)):
            if targetMask[j-lenInput] == 1:
                labels[i,j] = targetTokens[j-lenInput]

    tokenized_sequences['labels'] = labels
    #print(tokenized_sequences)
    #print(tokenized_sequences['labels'])
    
    return tokenized_sequences.to(DEVICE)


def add_prefixes(x: List[str], y: List[str], dataset: str) -> Tuple[List[str], List[str]]:
    input_prefix = '' if utils.is_qa_dataset(dataset) else ''
    label_prefix = ' In the' if utils.is_qa_dataset(dataset) else ' TL;DR:'
    label_suffix = '.' if utils.is_qa_dataset(dataset) else ''

    x = [input_prefix + x_.replace('\n', ' ') + label_prefix for x_ in x]
    y = [' ' + y_.replace('\n', ' ') + label_suffix for y_ in y]

    return x, y


def ft_gpt2(model, tokenizer, x, y, mode, dataset, batch_size=8, grad_accum=8):
    x, y = add_prefixes(x, y, dataset)
    model = copy.deepcopy(model)

    # Debug print
    #print("Input x requires_grad:", any(isinstance(item, torch.Tensor) and item.requires_grad for item in x))
    #print("Input y requires_grad:", any(isinstance(item, torch.Tensor) and item.requires_grad for item in y))

    if mode.startswith('lora'):
        for m in model.transformer.h:
            m.mlp.c_fc = LoRAConv1DWrapper(m.mlp.c_fc, int(mode[4:]))
            m.mlp.c_proj = LoRAConv1DWrapper(m.mlp.c_proj, int(mode[4:]))
            m.attn.c_attn = LoRAConv1DWrapper(m.attn.c_attn, int(mode[4:]))

    model.to(DEVICE)
    optimizer = torch.optim.Adam(parameters_to_fine_tune(model, mode), lr=2e-5)

    # Debug print
    print(f"Parameters with requires_grad=True for mode '{mode}':")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}")

    # Debug print
    print("Number of parameters with requires_grad=True:", sum(p.requires_grad for p in model.parameters()))

    all_both = tokenize_gpt2_batch(tokenizer, x, y)
    max_n = len(x) * 10
    pbar = tqdm.tqdm(range(max_n))
    idxs = []
    for step in pbar:
        model.train()

        if len(idxs) < batch_size // grad_accum:
            idxs = list(range(len(x)))
            random.shuffle(idxs)
        batch_idxs = idxs[:batch_size // grad_accum]
        idxs = idxs[batch_size // grad_accum:]

        x_batch = [x[i] for i in batch_idxs]
        y_batch = [y[i] for i in batch_idxs]

        batch = tokenize_gpt2_batch(tokenizer, x_batch, y_batch)
        
        # Debug print
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"{key} requires_grad:", value.requires_grad)

        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name} requires gradient")


        model.train()  # Set the model to training mode
        model_output = model(**batch, use_cache=False)
        loss = get_loss(model_output.logits, batch['labels'])
        loss = loss / grad_accum

        # Debug print
        print(f"Loss value: {loss.item()}")
        print(f"Loss requires grad: {loss.requires_grad}")
        print(f"Loss has grad_fn: {loss.grad_fn is not None}")

        # Check if any model parameters require gradients
        params_require_grad = any(p.requires_grad for p in model.parameters())
        print(f"Any model parameters require grad: {params_require_grad}")

        # If using an optimizer, check its param groups
        if optimizer.param_groups:
            optim_params_require_grad = any(p.requires_grad for group in optimizer.param_groups for p in group['params'])
            print(f"Any optimizer parameters require grad: {optim_params_require_grad}")

        # Check individual layers
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name} requires grad and has size {param.size()}")

        loss.backward()

        # Debug print
        print("Gradients after backward:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.grad is not None}")

        if (step + 1) % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad()

            # Debug print
            print("Parameter update check:")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(f"{name} updated: {param.grad is not None}")

        pbar.set_postfix(loss=loss.item())

        if step % (grad_accum * 5) == 0:
            with torch.inference_mode():
                model.eval()
                accs = []
                for idx in range(len(list(all_both.values())[0])):
                    d = {k: v[idx:idx+1] for k, v in all_both.items()}
                    acc = get_acc(model(**d).logits, d['labels'])
                    accs.append(acc)
                total_acc = sum(accs) / len(accs)
                pbar.set_description(f'Fine-tuning acc: {total_acc:.04f}')

            if total_acc >= utils.early_stop_thresold(dataset):
                print('Early stopping!')
                break

        if step == 1:
            break

    return model


def eval(model, tokenizer, val_data):
    x = tokenizer(val_data['x'], return_tensors='pt', padding=True, truncation=True, max_length=100).to(DEVICE)
    y = torch.tensor(val_data['y'], device=DEVICE)
    with torch.inference_mode():
        logits = model(**x).logits
    return get_acc(logits, y)


def run_ft(models: List[str], datasets: List[str], ks: List[int], modes: List[str], debug: bool, repeats: int, n_val: int = 125):
    results = {}

    for dataset in datasets:
        utils.fix_random_seeds()

        if debug:
            n_val = 1   
        train, val = utils.get_dataset(dataset, max(ks), n_val=n_val)

        for model_name, mode in itertools.product(models, modes):
            utils.fix_random_seeds()
            
            if dataset == 'amazon':
                model, tokenizer = utils.get_model_and_tokenizer(model_name, transformers.AutoModelForSequenceClassification, num_labels=5)
            else:
                model, tokenizer = utils.get_model_and_tokenizer(model_name, transformers.AutoModelForCausalLM)

            stop_tokens = utils.stop_tokens(tokenizer)

            for k in ks:
                print(f'Fine-tuning {model_name} on {dataset} with k={k} and mode={mode}')

                utils.fix_random_seeds()

                for repeat in range(repeats):
                    result_key = '_'.join([model_name, dataset, str(k), mode])

                    if repeat > 0:
                        print(f'Beginning repeat #{repeat}')

                    if dataset == 'amazon':
                        fine_tuned = ft_bert(model, tokenizer, train['x'][:k*5], train['y'][:k*5], mode, debug)
                        val_acc = eval(fine_tuned, tokenizer, val)

                        if val_acc >= results.get(result_key, 0):
                            results[result_key] = val_acc

                    else:
                        if k > 0:
                            fine_tuned = ft_gpt2(model, tokenizer, train['x'][:k], train['simple_y'][:k], mode, dataset)
                        else:
                            fine_tuned = copy.deepcopy(model)
                            fine_tuned.to(DEVICE)

                        fine_tuned.eval()
                        targets = []
                        predictions = []
                        pbar = tqdm.tqdm(list(range(min(n_val, len(val['x'])))))

                        for row in pbar:
                            test_input = val['x'][row]
                            targets.append(val['y'][row])
                            max_tokens = utils.max_sampled_tokens_for_dataset(dataset)
                            prompt_mode = 'qa' if utils.is_qa_dataset(dataset) else 'tldr'
                            prompt = get_icl_prompts([], [], test_input, prompt_mode=prompt_mode)
                            input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(DEVICE)
                            sampled_tokens = do_sample(fine_tuned, input_ids, stop_tokens, max_tokens)
                            decoded = tokenizer.decode(sampled_tokens).strip()
                            predictions.append(decoded)
                            metric = get_performance_metric(predictions, targets, utils.metric_for_dataset(dataset))
                            pbar.set_description(f'Eval: {metric:.04f}')

                        if metric >= results.get(result_key, 0):
                            results[result_key] = metric
                        results['_'.join([model_name, dataset, str(k), mode])] = metric

                    print(results)
                    question = 'ft'
                    if not os.path.exists(f'results/{question}'):
                        os.makedirs(f'results/{question}')

                    for k_, v in results.items():
                        with open(f'results/{question}/{k_}.json', 'w') as f:
                            json.dump({'metric': v}, f)
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