#!/bin/bash

pip install -q -U trl transformers accelerate git+https://github.com/huggingface/peft.git
pip install -q datasets bitsandbytes einops wandb

huggingface-cli login

cd ..
cd ..

python3 finetune_HF_model.py \
  --model_name "tiiuae/falcon-7b-instruct" \
  --run_name 'falcon-7b-instruct-lora-metaphor-nonCoT' \
  --dataset 'JinchengLiu2000/Metaphot_non_CoT_ft' \
  --peft_name 'falcon-7b-instruct-lora-metaphor-nonCoT' \
  --output_dir 'falcon-7b-instruct-lora-metaphor-nonCoT-results'