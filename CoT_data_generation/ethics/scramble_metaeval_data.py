import json

from pathlib import Path
import random


def scramble(
    source_path:Path = Path("prepare_ethics_CoT_dataset/data"),
    commonsense_CoT_data_file_name:str = "ethics_commonsense_CoT_reasoning.json",
    deontology_CoT_data_file_name:str = "ethics_deontology_CoT_reasoning.json",
    justice_CoT_data_file_name:str = "ethics_justice_CoT_reasoning.json",
    CoT_destination_path:Path = Path("prepare_ethics_CoT_dataset/data/CoT"),
    CoT_out_file_name:str = "ethics_CoT_reasoning_scrambled.json",
    commonsense_non_CoT_data_file_name:str = "ethics_commonsense_non_CoT_reasoning.json",
    deontology_non_CoT_data_file_name:str = "ethics_deontology_non_CoT_reasoning.json",
    justice_non_CoT_data_file_name:str = "ethics_justice_non_CoT_reasoning.json",
    non_CoT_destination_path:Path = Path("prepare_ethics_CoT_dataset/data/non_CoT"),
    non_CoT_out_file_name:str = "ethics_non_CoT_reasoning_scrambled.json",
    

) ->None:
    commonsense_CoT_data_path = source_path / commonsense_CoT_data_file_name
    with open(commonsense_CoT_data_path, mode = "r") as file:
        commonsense_CoT_data = json.load(file)

    deontology_CoT_data_path = source_path / deontology_CoT_data_file_name
    with open(deontology_CoT_data_path, mode = "r") as file:
        deontology_CoT_data = json.load(file)

    justice_CoT_data_path = source_path / justice_CoT_data_file_name
    with open(justice_CoT_data_path, mode = "r") as file:
        justice_CoT_data = json.load(file)
    CoT_json_list = commonsense_CoT_data
    CoT_json_list.extend(deontology_CoT_data)
    CoT_json_list.extend(justice_CoT_data)


    commonsense_non_CoT_data_path = source_path / commonsense_non_CoT_data_file_name
    with open(commonsense_non_CoT_data_path, mode = "r") as file:
        commonsense_non_CoT_data = json.load(file)

    deontology_non_CoT_data_path = source_path / deontology_non_CoT_data_file_name
    with open(deontology_non_CoT_data_path, mode = "r") as file:
        deontology_non_CoT_data = json.load(file)

    justice_non_CoT_data_path = source_path / justice_non_CoT_data_file_name
    with open(justice_non_CoT_data_path, mode = "r") as file:
        justice_non_CoT_data = json.load(file)

    non_CoT_json_list = commonsense_non_CoT_data
    non_CoT_json_list.extend(deontology_non_CoT_data)
    non_CoT_json_list.extend(justice_non_CoT_data)

    shuffled_index = [i for i in range(0, len(CoT_json_list))];
    random.shuffle(shuffled_index)
    
    CoT_json_list_out = []
    non_CoT_json_list_out = []
    for indexes in shuffled_index:
        CoT_json_list_out.append(CoT_json_list[indexes])
        non_CoT_json_list_out.append(non_CoT_json_list[indexes])

    CoT_destination_file = CoT_destination_path / CoT_out_file_name
    CoT_json_object = json.dumps(CoT_json_list_out, indent=4)
    with open(CoT_destination_file, 'w') as file:
        file.write(CoT_json_object)

    non_CoT_destination_file = non_CoT_destination_path / non_CoT_out_file_name
    non_CoT_json_object = json.dumps(non_CoT_json_list_out, indent=4)
    with open(non_CoT_destination_file, 'w') as file:
        file.write(non_CoT_json_object)
    
    return 


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(scramble)