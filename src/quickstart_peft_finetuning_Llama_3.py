#!/usr/bin/env python
# coding: utf-8

# ## PEFT Finetuning Quick Start
# 
# Train a Meta Llama 3 model on a single GPU (e.g. A10 with 24GB) using int8 quantization and LoRA finetuning.
# 
# **_Note:_** To run this code on a machine with less than 24GB VRAM (e.g. T4 with 16GB) the context length of the training dataset 
# needs to be adapted. We do this based on the available VRAM during execution. If you run into OOM issues try to further lower the value 
# of train_config.context_length.

# ### Step 0: Install pre-requirements and convert checkpoint 
# We need to have llama-recipes and its dependencies installed. Additionally, we need to log in with the huggingface_cli 
# and make sure that the account is able to to access the Meta Llama weights.

# ! pip install llama-recipes ipywidgets

import huggingface_hub
huggingface_hub.login()

# ### Step 1: Load the model 
# Setup training configuration and load the model and tokenizer.

import torch
from transformers import LlamaForCausalLM, AutoTokenizer
from llama_recipes.configs import train_config as TRAIN_CONFIG

# make a class for the training configuration
class TrainingConfig:
    def __init__(self):
        self.model_name = "meta-llama/Meta-Llama-3.1-8B"
        self.num_epochs = 1
        self.run_validation = False
        self.gradient_accumulation_steps = 4
        self.batch_size_training = 1
        self.lr = 3e-4
        self.use_fast_kernels = True
        self.use_fp16 = True
        self.context_length = 1024 if torch.cuda.get_device_properties(0).total_memory < 16e9 else 2048 # T4 16GB or A10 24GB
        self.batching_strategy = "packing"
        self.output_dir = "meta-llama-samsum"
        self.use_peft = True


# make a class for PEFT
class PEFTConfig:
    def __init__(self):
        self.r = 8
        self.lora_alpha = 32
        self.lora_dropout = 0.01
        

train_config = TrainingConfig()

from transformers import BitsAndBytesConfig
config = BitsAndBytesConfig(
    load_in_8bit=True,
)

model = LlamaForCausalLM.from_pretrained(
            train_config.model_name,
            device_map="auto",
            quantization_config=config,
            use_cache=False,
            attn_implementation="sdpa" if train_config.use_fast_kernels else None,
            torch_dtype=torch.float16,
        )

tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
tokenizer.pad_token = tokenizer.eos_token


# ### Step 2: Check base model 
# Run the base model on an example input:

eval_prompt = """
Summarize this dialog:
A: Hi Tom, are you busy tomorrow’s afternoon?
B: I’m pretty sure I am. What’s up?
A: Can you go with me to the animal shelter?.
B: What do you want to do?
A: I want to get a puppy for my son.
B: That will make him so happy.
A: Yeah, we’ve discussed it many times. I think he’s ready now.
B: That’s good. Raising a dog is a tough issue. Like having a baby ;-) 
A: I'll get him one of those little dogs.
B: One that won't grow up too big;-)
A: And eat too much;-))
B: Do you know which one he would like?
A: Oh, yes, I took him there last Monday. He showed me one that he really liked.
B: I bet you had to drag him away.
A: He wanted to take it home right away ;-).
B: I wonder what he'll name it.
A: He said he’d name it after his dead hamster – Lemmy  - he's  a great Motorhead fan :-)))
---
Summary:
"""

model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

#model.eval()
#with torch.inference_mode():
#    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))

# ### Step 3: Load the preprocessed dataset
# We load and preprocess the samsum dataset which consists of curated pairs of dialogs and their summarization:

from llama_recipes.configs.datasets import samsum_dataset
from llama_recipes.utils.dataset_utils import get_dataloader

samsum_dataset.trust_remote_code = True

train_dataloader = get_dataloader(tokenizer, samsum_dataset, train_config)
eval_dataloader = get_dataloader(tokenizer, samsum_dataset, train_config, "val")

# ### Step 4: Prepare model for PEFT
# Let's prepare the model for Parameter Efficient Fine Tuning (PEFT):

from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig
from dataclasses import asdict
from llama_recipes.configs import lora_config as LORA_CONFIG

lora_config = LORA_CONFIG()
lora_config.r = 8
lora_config.lora_alpha = 32
lora_dropout: float=0.01

peft_config = LoraConfig(**asdict(lora_config))

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

# ### Step 5: Fine tune the model
# Here, we fine tune the model for a single epoch.

import torch.optim as optim
from llama_recipes.utils.train_utils import train
from torch.optim.lr_scheduler import StepLR

model.train()

optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )
scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

# Start the training process
results = train(
    model,
    train_dataloader,
    eval_dataloader,
    tokenizer,
    optimizer,
    scheduler,
    train_config.gradient_accumulation_steps,
    train_config,
    None,
    None,
    None,
    wandb_run=None,
)

# ### Step 6:
# Save model checkpoint
model.save_pretrained(train_config.output_dir)


# ### Step 7:
# Try the fine tuned model on the same example again to see the learning progress:

model.eval()
with torch.inference_mode():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))

