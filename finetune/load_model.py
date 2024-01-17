import transformers
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

def load_model():
    model = transformers.AutoModelForCausalLM.from_pretrained(
        "togethercomputer/RedPajama-INCITE-Instruct-3B-v1",
        device_map={"": 0}
    )
    return 

if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    from jsonargparse import CLI
    load_model()
    
    