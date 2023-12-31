#!/bin/bash

mkdir data
mkdir data/CoT
mkdir data/non_CoT

cd ..
python3 stock_market_sentiment_analysis/generate_financial_news_CoT_data.py \
  --api_key "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" \
  --destination_path_CoT "stock_market_sentiment_analysis/data/CoT" \
  --destination_path_non_CoT "stock_market_sentiment_analysis/data/non_CoT" \
  --CoT_out_file_name "financial_news_CoT_reasoning_0.json" \
  --non_CoT_out_file_name "financial_news_non_CoT_reasoning_0.json" \
  --start_index 0 \
  --end_index 3000

python3 stock_market_sentiment_analysis/generate_financial_news_CoT_data.py \
  --api_key "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" \
  --destination_path_CoT "stock_market_sentiment_analysis/data/CoT" \
  --destination_path_non_CoT "stock_market_sentiment_analysis/data/non_CoT" \
  --CoT_out_file_name "financial_news_CoT_reasoning_1.json" \
  --non_CoT_out_file_name "financial_news_non_CoT_reasoning_1.json" \
  --start_index 3000 \
  --end_index 6000

python3 stock_market_sentiment_analysis/generate_financial_news_CoT_data.py \
  --api_key "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" \
  --destination_path_CoT "stock_market_sentiment_analysis/data/CoT" \
  --destination_path_non_CoT "stock_market_sentiment_analysis/data/non_CoT" \
  --CoT_out_file_name "financial_news_CoT_reasoning_2.json" \
  --non_CoT_out_file_name "financial_news_non_CoT_reasoning_2.json" \
  --start_index 6000 \
  --end_index 9000

python3 stock_market_sentiment_analysis/generate_financial_news_CoT_data.py \
  --api_key "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" \
  --destination_path_CoT "stock_market_sentiment_analysis/data/CoT" \
  --destination_path_non_CoT "stock_market_sentiment_analysis/data/non_CoT" \
  --CoT_out_file_name "financial_news_CoT_reasoning_3.json" \
  --non_CoT_out_file_name "financial_news_non_CoT_reasoning_3.json" \
  --start_index 9000 \
  --end_index 12000

python3 stock_market_sentiment_analysis/merge_data.py \
  --destination_path_CoT "stock_market_sentiment_analysis/data/CoT" \
  --destination_path_non_CoT "stock_market_sentiment_analysis/data/non_CoT" \
  --CoT_out_file_name "financial_news_CoT_reasoning.json" \
  --non_CoT_out_file_name "financial_news_non_CoT_reasoning.json" 

