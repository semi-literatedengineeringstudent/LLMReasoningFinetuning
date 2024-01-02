#!/bin/bash

mkdir data


cd ..


python3 bias/data_generator.py \
  --api_key "your_api_key" \
  --destination_path "metaphor/data" \
  --out_file_name "Metaphor_CoT_explanation" \
  --out_file_name_non_CoT "Metaphor_non_CoT"