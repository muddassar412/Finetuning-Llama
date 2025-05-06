'''
!pip install peft
!pip install accelerate
!pip install bitsandBytes
!pip install transformers
!pip install datasets
!pip install GPUtil
'''

import torch
import GPUtil
import os


import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, LlamaTokenizer
from huggingface_hub import notebook_login
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model



GPUtil.showUtilization()

if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is not available, using CPU instead")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"




if "COLAB_GPU" in os.environ:
  from google.colab import output
  output.enable_custom_widget_manager()


if "COLAB_GPU" in os.environ:
  !huggingface-cli login
else:
  notebook_login()


base_model_id = "meta-llama/Llama-2-7b-chat-hf"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config)


#git Clone
!git clone https://github.com/poloclub/Fine-tuning-LLMs.git

train_dataset = load_dataset("text", data_files={"train":
                                                 ["/content/Fine-tuning-LLMs/data/hawaii_wf_4.txt", "/content/Fine-tuning-LLMs/data/hawaii_wf_2.txt"]}, split="train")

train_dataset["text"][1]

tokenizer = LlamaTokenizer.from_pretrained(base_model_id, use_fast=False, trust_remote_code=True, add_eos_token=True)

if tokenizer.pad_token is None:
  tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})


tokenized_train_dataset = []
for phrase in train_dataset:
  tokenized_train_dataset.append(tokenizer(phrase["text"]))

tokenized_train_dataset[1]


tokenized_train_dataset[2]

tokenizer.eos_token

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=8,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)


trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    args=transformers.TrainingArguments(
        output_dir="./finetunedModel",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        learning_rate=1e-4,
        max_steps=20,
        bf16=False,
        optim="paged_adamw_8bit",
        logging_dir="./log",
        save_strategy="epoch",
        save_steps=50,
        logging_steps=10

),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache=False
trainer.train()


base_model_id = "meta-llama/Llama-2-7b-chat-hf"

nf4Config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = LlamaTokenizer.from_pretrained(base_model_id, use_fast=False, trust_remote_code=True, add_eos_token=True)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=nf4Config,
    device_map="auto",
    trust_remote_code=True,
    use_auth_token=True
  )



tokenizer = LlamaTokenizer.from_pretrained(base_model_id, use_fast=False, trust_remote_code=True, add_eos_token=True

                              )

modelFinetuned = PeftModel.from_pretrained(base_model, "finetunedModel/checkpoint-20")


user_question = "When did Hawaii wildfires start?"

eval_prompt = f"Question: {user_question} Just answer this question accurately and concisely.\n"

promptTokenized = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

modelFinetuned.eval()

with torch.no_grad():
  print(tokenizer.decode(modelFinetuned.generate(**promptTokenized, max_new_tokens=1024)[0], skip_special_tokens=True))
  torch.cuda.empty_cache()

