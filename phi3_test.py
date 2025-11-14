# Using Phi3 LLM in local.
# install torch>=2.4.0 transformers>=4.53.0 sentencepiece>=0.2

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# download and save in local.
model_id = "microsoft/Phi-3-mini-128k-instruct"
save_path = "./Phi3"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

#save locally.
Tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

#Loading model.
local_path = "./Phi3"

#Quantization config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True
    )

tokenizer = AutoTokenizer.from_pretrained(local_path)
model = AutoModelForCausalLM.from_pretrained(
    local_path,
    #quantization_config=quant_config,
    device_map='auto'
    )

#Inference
messages [
    {'role':'system', 'content':'You are a helpful assistant.'},
    {'role':'user', 'content':'Give me three bullet points on vector database.'},
    ]

text = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors='pt').to(model.device)

outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    max_new_tokens=1024
    )

response = tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)


