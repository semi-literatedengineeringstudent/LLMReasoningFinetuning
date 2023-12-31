import json
from pathlib import Path

import os
 
def combine_all_data(destination_path_CoT:Path = Path("stock_market_sentiment_analysis/data/CoT"), 
                      destination_path_non_CoT:Path = Path("stock_market_sentiment_analysis/data/non_CoT"),
                      CoT_out_file_name:str = "financial_news_CoT_reasoning.json",
                      non_CoT_out_file_name:str = "financial_news_non_CoT_reasoning.json" 
                      )->None:
    
    CoT_data_total = []
    for json_file in os.listdir(destination_path_CoT):
        file_location = destination_path_CoT / json_file
        with open(file_location, "r") as f:
            CoT_data_total.extend(json.load(f))
            
    destination_CoT_json = destination_path_CoT / CoT_out_file_name

    with open(destination_CoT_json, "w") as f:
        json.dump(CoT_data_total, f, indent=4)

    non_CoT_data_total = []
    for json_file in os.listdir(destination_path_non_CoT):
        file_location = destination_path_non_CoT / json_file
        with open(file_location, "r") as f:
            non_CoT_data_total.extend(json.load(f))
    destination_non_CoT_json = destination_path_non_CoT / non_CoT_out_file_name

    with open(destination_non_CoT_json, "w") as f:
        json.dump(non_CoT_data_total, f, indent=4)
    
    return
    


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    from jsonargparse import CLI
    
    CLI(combine_all_data)