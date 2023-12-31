from datasets import load_dataset
import json
from openai import OpenAI
from pathlib import Path
from tqdm import tqdm
import queue
import threading
import csv
import random 


def process_metaeval_deontology_data(deontology_row:dict,
                     client:OpenAI,
                     CoT_queue:queue,
                     non_CoT_queue:queue)->None:
    try:
        reasonable = deontology_row["label"]  
        scenario = deontology_row["scenario"]
        excuse = deontology_row["excuse"]
 
        CoT_new_row_deontology = {}
        system_prompt_stereotype = ""
        user_prompt_stereotype = ""
        if (reasonable == '0'):
            (system_prompt_stereotype, user_prompt_stereotype) = generate_prompt_unreasonable(scenario, excuse)
        else:
            (system_prompt_stereotype, user_prompt_stereotype) = generate_prompt_reasonable(scenario, excuse)

        response_deontology = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt_stereotype},
                {"role": "user", "content": user_prompt_stereotype},
                ]
            )
    
        response_deontology_content = response_deontology.choices[0].message.content
    
        CoT_new_row_deontology["instruction"] = f"""Determine if the following excuse/statement is reasonable given the provided scenario/request or not. Provide a Chain of Thought reasoning to justify your answer."""
        CoT_new_row_deontology["input"] = f"""
        scenario/request: {scenario}
        
        excuse/statement: {excuse}
        """
        CoT_new_row_deontology["output"] = response_deontology_content

        non_CoT_new_row_deontology = {}
        non_CoT_new_row_deontology["instruction"] = f"""Determine if the following excuse/statement is reasonable given the provided scenario/request or not. If it is reasonable, reply "True", otherwise reply "False"."""
        non_CoT_new_row_deontology["input"] = f"""
        scenario/request: {scenario}
        
        excuse/statement: {excuse}
        """
        if (reasonable == '0'):
            non_CoT_new_row_deontology["output"] = "False"
        else:
            non_CoT_new_row_deontology["output"] = "True"

        CoT_queue.put(CoT_new_row_deontology)
        non_CoT_queue.put(non_CoT_new_row_deontology)
        
    except Exception as e:
        print(f"Error:{e}")
        CoT_queue.put(None)
        non_CoT_queue.put(None)

def generate_prompt_reasonable(
    scenario:str,
    excuse:str,
)->(str, str):

    system_prompt = f"""You are a language model that can make good deontological judgement that you can tell the right or wrong of a statement/action given scenario/request."""
    user_prompt = f"""The following content contains an scenario/request: {scenario}.
    For this scenario/request, there is an excuse/statement: {excuse} 
    The excuse/statement is deemed acceptable.
    Generate an Chain of Thought reasoning on why the excuse/statement is reasonable given the duties or constraints in scenario/request.
    Start your response with "The excuse/statement given scenario/request is deemed acceptable." and Chain of Thought reasoning following it without explicitly mentioing you are applying Chain of Thought reasoning."""
    return (system_prompt, user_prompt)
    
def generate_prompt_unreasonable(
    scenario:str,
    excuse:str,
)->(str, str):
    system_prompt = f"""You are a language model that can make good deontological judgement that you can tell the right or wrong of a statement/action given scenario/request."""
    user_prompt = f"""The following content contains an scenario/request: {scenario}.
    For this scenario/request, there is an excuse/statement: {excuse} 
    The excuse/statement is deemed ethically unaccepatble
    Generate an Chain of Thought reasoning on why the excuse/statement is unreasonable given the duties or constraints in scenario/request.
    Start your response with "The excuse/statement given scenario/request is deemed unacceptable." and Chain of Thought reasoning following it without explicitly mentioing you are applying Chain of Thought reasoning."""
    return (system_prompt, user_prompt)

def generate_CoT_From_GPT(
    api_key:str = "",
    data_file_path:Path = Path("/prepare_ethics_CoT_dataset/ethics_raw_data/deontology/cm_deontology.csv"),
    destination_path:Path = Path("prepare_ethics_CoT_dataset/data"),
    CoT_out_file_name:str = "ethics_deontology_CoT_reasoning.json",
    non_CoT_out_file_name:str = "ethics_deontology_non_CoT_reasoning.json",
    sample_number:int = 2000,
    acceptable_unacceptable_ratio:float = 0.5,
) -> None:
    raw_dataset = None
    with open(data_file_path, mode = "r", encoding = 'utf-8') as file:
        csv_reader = csv.DictReader(file)
        raw_dataset = list(csv_reader)
    
    assert sample_number <= len(raw_dataset), "number samples to pick from raw dataset should be lower or equal to the size of entire raw dataset."
    data_acceptable = []
    data_unacceptable = []
    for data_rows in raw_dataset:
        if (data_rows["label"] == '0'):
            data_acceptable.append(data_rows)
        else:
            data_unacceptable.append(data_rows)
    acceptable_size = int(sample_number * acceptable_unacceptable_ratio)
    unacceptable_size = int(sample_number - acceptable_size)

    index_to_pick_acceptable = random.sample(range(len(data_acceptable)), acceptable_size)
    index_to_pick_unacceptable = random.sample(range(len(data_unacceptable)), unacceptable_size)
    dataset = []
    for indexes in index_to_pick_acceptable:
        dataset.append(data_acceptable[indexes])
    for indexes in index_to_pick_unacceptable:
        dataset.append(data_unacceptable[indexes])
    random.shuffle(dataset)
    client = OpenAI(api_key = api_key)
    
    CoT_queue = queue.Queue()
    non_CoT_queue = queue.Queue()
    threads = []

    for i in tqdm(range(0, len(dataset)), desc = "Number of samples evaluated:"):
        deontology_row = dataset[i]
        thread = threading.Thread(target = process_metaeval_deontology_data, args = (deontology_row, client, CoT_queue, non_CoT_queue))
        threads.append(thread)

    for thread in threads:
        thread.start()
 
    for thread in threads:
        thread.join()

    CoT_json_list_total = []
    non_CoT_json_list_total = []
    while not CoT_queue.empty():
        CoT_reponse = CoT_queue.get()
        if CoT_reponse is not None:
            CoT_json_list_total.append(CoT_reponse)

    while not non_CoT_queue.empty():
        non_CoT_reponse = non_CoT_queue.get()
        if non_CoT_reponse is not None:
            non_CoT_json_list_total.append(non_CoT_reponse)


    CoT_json_list_total_string = json.dumps(CoT_json_list_total, indent=4)
    out_file_path_CoT_json_list_total_string = destination_path / (CoT_out_file_name)
    with open(out_file_path_CoT_json_list_total_string, 'w') as file:
        file.write(CoT_json_list_total_string)

    non_CoT_json_list_total_string = json.dumps(non_CoT_json_list_total, indent=4)
    out_file_path_non_CoT_json_list_total_string = destination_path / (non_CoT_out_file_name)
    with open(out_file_path_non_CoT_json_list_total_string, 'w') as file:
        file.write(non_CoT_json_list_total_string)
      
    return

    


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    from jsonargparse import CLI
    
    CLI(generate_CoT_From_GPT)