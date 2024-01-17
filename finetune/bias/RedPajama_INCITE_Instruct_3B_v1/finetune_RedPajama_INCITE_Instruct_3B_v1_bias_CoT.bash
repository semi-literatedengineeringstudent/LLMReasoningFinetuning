#!/bin/bash

pip install -q -U trl transformers accelerate git+https://github.com/huggingface/peft.git
pip install -q datasets bitsandbytes einops wandb

huggingface-cli login

cd ..
cd ..

python3 finetune_HF_model.py \
  --model_name "togethercomputer/RedPajama-INCITE-Instruct-3B-v1" \
  --run_name 'RedPajama-INCITE-Instruct-3B-v1-instruct-lora-bias-CoT' \
  --dataset 'yc4142/bias-CoT' \
  --peft_name 'RedPajama-INCITE-Instruct-3B-v1-instruct-lora-bias-CoT' \
  --output_dir 'RedPajama-INCITE-Instruct-3B-v1-instruct-lora-bias-CoT-results'