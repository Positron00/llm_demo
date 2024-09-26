from typing import List

import transformers
import utils
import itertools

def ft_llm():
    pass

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

