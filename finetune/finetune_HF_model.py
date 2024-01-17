import json
import sys
from pathlib import Path
from typing import Optional
from transformers import AutoTokenizer
import requests
from datasets import Dataset
from datasets import load_dataset
from tqdm import tqdm

import huggingface_hub

import transformers
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer

def generate_prompt(example) -> str:
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""
    output_texts = []
    for i in range(len(example["instruction"])):
      if example["input"]:
        output_texts.append((
              "Below is an instruction that describes a task, paired with an input that provides further context. "
              "Write a response that appropriately completes the request.\n\n"
              f"### Instruction:\n{example['instruction'][i]}\n\n### Input:\n{example['input'][i]}\n\n### Response: {example['output'][i]}" + "<eos>"
          ))
        output_texts.append((
          "Below is an instruction that describes a task. "
          "Write a response that appropriately completes the request.\n\n"
          f"### Instruction:\n{example['instruction'][i]}\n\n### Response:" + "<eos>"
      ))
    return output_texts;


def finetune_model(model_name:str = "tiiuae/falcon-7b-instruct",
                   run_name:str = 'falcon-7b-instruct-lora-bias-CoT',
                   dataset:str = 'yc4142/bias-CoT',
                   peft_name:str = 'falcon-7b-instruct-lora-bias-CoT',
                   output_dir:str = 'falcon-7b-instruct-lora-bias-CoT-results') -> None:
   report_to = "wandb" 
   if report_to != "none":
    import wandb
    wandb.login()
   wandb.init(project=run_name,config={
    "model": model_name,
    "dataset":dataset
    })
   repo_id = f'{huggingface_hub.whoami()["name"]}/{peft_name}'
   print(repo_id) 
   print("Loading model for model: ", model_name)

   model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": 0}
    )
   
   lora_config = LoraConfig(
    r= 8,
    lora_alpha=16,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
   )
   model = get_peft_model(model, lora_config)
   model.print_trainable_parameters()
   eval_steps = 100
   save_steps = 100
   logging_steps = 20
   
   split_dataset = load_dataset(dataset, split='train').train_test_split(test_size=0.03865, seed = 42)
   train_data = split_dataset['train']
   eval_data = split_dataset['test']
   tokenizer = AutoTokenizer.from_pretrained(model_name[0],add_eos_token=True)
   tokenizer.pad_token_id = 0
   tokenizer.add_special_tokens({'eos_token':'<eos>'})

   trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset= eval_data,
    formatting_func=generate_prompt,
    max_seq_length = 2048,
    args=transformers.TrainingArguments(
        num_train_epochs=5,
        learning_rate=3e-4,
        logging_steps=logging_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        output_dir=output_dir,
        report_to=report_to if report_to else "none",
        save_total_limit=3,
        load_best_model_at_end=True,
        push_to_hub=False,
        auto_find_batch_size=True,
        warmup_steps=100,
        weight_decay = 0.01,
        per_device_train_batch_size = 2,
        per_device_eval_batch_size = 2,
        gradient_accumulation_steps = 32
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
   model.config.use_cache = False

   trainer.train()

   trainer.model.push_to_hub(repo_id)
   tokenizer.push_to_hub(repo_id)
   
   return



if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    from jsonargparse import CLI
    
    CLI(finetune_model)