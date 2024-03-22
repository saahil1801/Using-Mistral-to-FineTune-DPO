from auto_gptq import exllama_set_max_input_length
import torch
from datasets import Dataset, load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training,PeftModel
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, GPTQConfig , BitsAndBytesConfig
from trl import DPOTrainer
from transformers import GenerationConfig
from data_preprocessing import create_dataset
import warnings
warnings.filterwarnings("ignore")
import logging
from config import Config
from datetime import datetime


class MistralDPOTrainer:
    def __init__(self,config: Config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_ID)
        logging.info('Loaded tokenizer')
        if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = tokenizer.eos_token

    def create_double_dataset(self):
        dataset = create_dataset(self.config.DATASET_ID, split='train_prefs')
        df = dataset.to_pandas()    
        train_size = int(len(df) * 0.8)
        train_df = df[:train_size].sample(1000)
        train_dataset = Dataset.from_pandas(train_df)
        val_df = df[train_size:].sample(200)
        val_dataset = Dataset.from_pandas(val_df)
        return train_dataset, val_dataset
    
    def prepare_model(self):
        gptq_config = GPTQConfig(bits=self.config.BITS, use_exllama=self.config.DISABLE_EXLLAMA)
        model = AutoModelForCausalLM.from_pretrained(config.MODEL_ID, torch_dtype=torch.float16,
                                                     low_cpu_mem_usage=config.LOW_CPU_MEM_USAGE,
                                                     quantization_config=gptq_config,
                                                      device_map=self.config.DEVICE_MAP)

        model = exllama_set_max_input_length(model, max_input_length=4096)
        
        logging.info('Downloaded Model')

        model_ref=AutoModelForCausalLM.from_pretrained(config.MODEL_ID, torch_dtype=torch.float16,
                                                     low_cpu_mem_usage=config.LOW_CPU_MEM_USAGE,
                                                     quantization_config=gptq_config,
                                                      device_map=self.config.DEVICE_MAP)

        model_ref = exllama_set_max_input_length(model_ref, max_input_length=4096)
        logging.info('Downloaded Model_Reference')

        peft_config = LoraConfig(
            r=self.config.LORA_R,
            lora_alpha=self.config.LORA_ALPHA,
            lora_dropout=self.config.LORA_DROPOUT,
            target_modules=self.config.LORA_TARGET_MODULES,
            task_type=self.config.LORA_TASK_TYPE,
            bias=self.config.LORA_BIAS,
            inference_mode=self.config.INFERENCE_MODE)
        
        logging.info('Peft Config Initialized')

        model = prepare_model_for_kbit_training(model)
        model.config.use_cache=False
        model.gradient_checkpointing_enable()
        model.config.pretraining_tp=1
        model = get_peft_model(model, peft_config)

        model_ref = prepare_model_for_kbit_training(model_ref)
        model_ref.config.use_cache=False
        model_ref.gradient_checkpointing_enable()
        model_ref.config.pretraining_tp=1
        model_ref = get_peft_model(model_ref, peft_config)

        return model, model_ref, peft_config
    
    def set_training_arguments(self):
         
        training_arguments = TrainingArguments(
        per_device_train_batch_size=self.config.BATCH_SIZE,
        max_steps=self.config.MAX_STEPS,
        remove_unused_columns=self.config.REMOVE_UNUSED_COLUMNS,
        gradient_accumulation_steps=self.config.GRAD_ACCUMULATION_STEPS,
        gradient_checkpointing=self.config.GRAD_CHECKPOINTING,
        learning_rate=self.config.LEARNING_RATE,
        evaluation_strategy=self.config.EVALUATION_STRATEGY,
        logging_first_step=self.config.LOGGING_FIRST_STEP,
        logging_steps=self.config.LOGGING_STEPS,   
        output_dir=self.config.OUTPUT_DIRUNS,
        optim=self.config.OPTIM,
        warmup_steps=self.config.WARMUP_STEPS,
        fp16=self.config.FP16
        )
         
        return training_arguments
    
    def train(self):
        train_dataset, val_dataset = self.create_double_dataset()
        print('triple dataset for DPO', '*'*20)
        print('train_dataset', train_dataset)
        print('val_dataset', val_dataset)

        model, model_ref, peft_config = self.prepare_model()
        
        training_args = self.set_training_arguments()
        logging.info('Training arguments are set')
        dpo_trainer = DPOTrainer(
            model,
            model_ref,
            args=training_args,
            beta=0.1,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            max_length=256,
            max_target_length=128,
            max_prompt_length=128
        )
        dpo_trainer.train()
        logging.info('Model is trained')
        dpo_trainer.save_model(self.config.OUTPUT_MODEL_DIR)
        logging.info('Model is saved')

if __name__ == '__main__':
    config = Config()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log_filename = f"/teamspace/studios/this_studio/log/training/data_{timestamp}.log"
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_filename, filemode='w') 
        
    dpo_trainer = MistralDPOTrainer(config)
    dpo_trainer.train()
    
      
