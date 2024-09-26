import argparse
import utils
import ft
import transformers

# parse the input argument
parser = argparse.ArgumentParser()
parser.add_argument('--task')
parser.add_argument('--model')
parser.add_argument('--dataset')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--repeats', default=1, type=int)
args = parser.parse_args()

# download models and datasets
def cache():
    models = [
        {'name': 'Llama-3.1-8B', 'class': transformers.AutoModelForCausalLM},
        {'name': 'Llama-3.2-11B-V-Inst', 'class': transformers.MllamaForConditionalGeneration}
    ]
    
    for model in models:
        utils.get_model_and_tokenizer(model['name'], model['class'])

    datasets = [
        {'name': 'amazon', 'n_train': 1, 'n_val': 125},
        {'name': 'xsum', 'n_train': 8, 'n_val': 125}
    ]

    for dataset in datasets:
        utils.get_dataset(dataset=dataset['name'], n_train=dataset['n_train'], n_val=dataset['n_val'])

# run a task
def run():
    ks = [int(k) for k in args.k.split(',')]
    if args.task == 'run_ft':
        ft.run_ft(args.model.split(','), args.dataset.split(','), ks, args.mode.split(','), args.debug, args.repeats)
    
    elif args.task == 'cache':
        cache()

if __name__ == '__main__':
    run()