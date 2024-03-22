from transformers import GenerationConfig , AutoModelForCausalLM
from transformers import AutoTokenizer
import torch
import warnings
warnings.filterwarnings("ignore")
from config import Config
import logging
from datetime import datetime


if __name__ == '__main__':
    config= Config()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log_filename = f"/teamspace/studios/this_studio/log/inference/data_{timestamp}.log"
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_filename, filemode='w') 
    

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(config.OUTPUT_MODEL_DIR,low_cpu_mem_usage=config.LOW_CPU_MEM_USAGE,
        return_dict=config.RETURN_DICT,
        torch_dtype=torch.float16,
        device_map=config.DEVICE_MAP)

    generation_config = GenerationConfig(
        do_sample=config.DO_SAMPLE,
        top_k=config.TOP_K,
        temperature=config.TEMPERATURE,
        max_new_tokens=config.MAX_NEW_TOKENS2,
        pad_token_id=tokenizer.eos_token_id
    )
    logging.info('Test Input Loaded and processed')
    inputs=tokenizer(config.PROMPT, return_tensors='pt').to('cuda')
    outputs = model.generate(**inputs, generation_config=generation_config)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))