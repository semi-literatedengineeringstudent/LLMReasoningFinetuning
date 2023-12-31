from datasets import load_dataset
import json
from openai import OpenAI
from pathlib import Path
from tqdm import tqdm
import queue
import threading


def process_financial_news_data(financial_news_row:dict,
                     client:OpenAI,
                     CoT_queue:queue,
                     non_CoT_queue)->None:
    try:
        label = financial_news_row["label"]
        context = financial_news_row["text"]
        expected_price_change = ""
        sentiment = ""
        if (label == 0):
            expected_price_change = "fall"
            sentiment = "bearish"
        elif (label == 1):
            expected_price_change = "rise"
            sentiment = "bullish"
        else:
            expected_price_change = "stay constant"
            sentiment = "neutral"
        (system_prompt_financial_news, user_prompt_financial_news) = generate_prompt_financial_news(sentiment, context, expected_price_change)

         
        
        response_financial_news = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt_financial_news},
                {"role": "user", "content": user_prompt_financial_news},
                ]
            )
    
        response_financial_news_content = response_financial_news.choices[0].message.content.strip()

        CoT_financial_news_new_row = {}

        CoT_financial_news_new_row["instruction"] = f"""Identify people's perspective on stock market as a reaction to the following context. If you infer that people expect the stock price to go down, start your answer with "People will have bearish perspective about the stock." If you infer that people expect the stock price to go up, start your answer with "People will have bullish perspective about the stock." If you infer that people expect the stock price to stay constant, start your answer with "People will have neutral perspective about the stock." """
        CoT_financial_news_new_row["input"] = f"""
        context: {context}
        """
        CoT_financial_news_new_row["output"] = response_financial_news_content 
        CoT_queue.put(CoT_financial_news_new_row)

        non_CoT_financial_news_new_row = {}
        
        non_CoT_financial_news_new_row["instruction"] = f"""Identify people's perspective on stock market as a reaction to the following context. If you infer that people expect the stock price to go down, start your answer with "People will have bearish perspective about the stock." If you infer that people expect the stock price to go up, start your answer with "People will have bullish perspective about the stock." If you infer that people expect the stock price to stay constant, start your answer with "People will have neutral perspective about the stock." """
        non_CoT_financial_news_new_row ["input"] = f"""
        context: {context}
        """
        non_CoT_response = ""
        if (sentiment == "bearish"):
            non_CoT_response = "People will have bearish perspective about the stock."
        elif (sentiment == "bullish"):
            non_CoT_response = "People will have bullish perspective about the stock."
        else:
            non_CoT_response = "People will have neutral perspective about the stock."


        non_CoT_financial_news_new_row["output"] = non_CoT_response
        non_CoT_queue.put(non_CoT_financial_news_new_row)
        
    except Exception as e:
        print(f"Error:{e}")
        CoT_queue.put(None)
        non_CoT_queue.put(None)


def generate_prompt_financial_news(
    sentiment:str, 
    context:str,
    expected_price_change:str
)->(str, str):
    system_prompt = f"""You are an expert in stock market sentiment analysis. You understand how people would expect stock price to change due to some changes in market situation they learn from media such as title of financial news paper, why they will have such sentiment, and what they will do in the stock market due to their sentiment"""

    user_prompt = f"""The following text contains a title from a financial news paper: {context}. People react to the title with {sentiment} perspective. They will expect stock price to {expected_price_change}. Start your answer with "People will have {sentiment} perspective about the stock."
    Then, generate an Chain of Thought reasoning, explain why people would have {sentiment} perspective due to the title, identify what stock market concepts are involved and use those concept in your analysis, and what they would do given their expectation without explicitly mentioning you are applying Chain of Thought. Provide background on the company and industry involved in the title if it is necessary for justifying your chain of thought reasoning.."""
    return (system_prompt, user_prompt)
    





def generate_CoT_From_GPT(
    api_key:str = "",
    destination_path_CoT:Path = Path("prepare_bias_CoT_dataset/data"),
    destination_path_non_CoT:Path = Path("prepare_bias_CoT_dataset/data"),
    CoT_out_file_name:str = "",
    non_CoT_out_file_name:str = "",
    start_index:int = 0,
    end_index:int = 0
) -> None:
    dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")['train']
    client = OpenAI(api_key = api_key)
    
    CoT_queue = queue.Queue()
    non_CoT_queue = queue.Queue()
    threads = []
    if (end_index > len(dataset)):
        end_index = len(dataset)
    for i in tqdm(range(start_index, end_index), desc = "Number of samples evaluated:"):
        financial_news_row = dataset[i]
        thread = threading.Thread(target = process_financial_news_data, args = (financial_news_row, client, CoT_queue, non_CoT_queue))
        threads.append(thread)

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    CoT_json_list_total = []
    while not CoT_queue.empty():
        CoT_reponse = CoT_queue.get()
        if CoT_reponse is not None:
            CoT_json_list_total.append(CoT_reponse)

    CoT_json_list_total_string = json.dumps(CoT_json_list_total, indent=4)
    out_file_path_CoT_json_list_total_string = destination_path_CoT / CoT_out_file_name
    with open(out_file_path_CoT_json_list_total_string, 'w') as file:
        file.write(CoT_json_list_total_string)

    non_CoT_json_list_total= []
    while not non_CoT_queue.empty():
        non_CoT_reponse = non_CoT_queue.get()
        if non_CoT_reponse is not None:
            non_CoT_json_list_total.append(non_CoT_reponse)
            
    non_CoT_json_list_total_string = json.dumps(non_CoT_json_list_total, indent=4)
    out_file_path_non_CoT_json_list_total_string = destination_path_non_CoT / non_CoT_out_file_name
    with open(out_file_path_non_CoT_json_list_total_string, 'w') as file:
        file.write(non_CoT_json_list_total_string)
      
    return

    


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    from jsonargparse import CLI
    
    CLI(generate_CoT_From_GPT)