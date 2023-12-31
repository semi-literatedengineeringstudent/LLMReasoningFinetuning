#!/bin/bash

mkdir data


cd ..


python3 bias/generate_stereoset_multi_thread_updated.py \
  --api_key "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" \
  --destination_path "bias/data" \
  --CoT_out_file_name "bias_CoT_reasoning.json" \
  --non_CoT_out_file_name "bias_non_CoT_reasoning.json"


mkdir bias/data/CoT
mkdir bias/data/non_CoT

python3 bias/scramble_stereoset_data.py\
  --source_path "bias/data" \
  --CoT_data_file_name "bias_CoT_reasoning.json" \
  --non_CoT_data_file_name "bias_non_CoT_reasoning.json" \
  --CoT_destination_path "bias/data/CoT" \
  --CoT_out_file_name_train "bias_CoT_reasoning_scrambled_train.json" \
  --CoT_out_file_name_test "bias_CoT_reasoning_scrambled_test.json" \
  --non_CoT_destination_path "bias/data/non_CoT" \
  --non_CoT_out_file_name_train "bias_non_CoT_reasoning_scrambled_train.json" \
  --non_CoT_out_file_name_test "bias_non_CoT_reasoning_scrambled_test.json" \
  --test_ratio 0.03865
