from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_path = "./qlora-output/checkpoint-125/"

# Load base moderl (4-bit)
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_id)

# Inference
prompt = (
        "<|system|>\n"
        "You explain science simply and never make up facts.\n"
        "<|user|>\n"
        "Explain gravity to a 10-year-old using only real, simple facts. "
        "Do not mention numbers, speeds or formulas.\n"
        "<|assistant|>\n"
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=True,
            repetition_penalty=1.1,
            temperature=0.5,
            top_p=0.9,
            top_k=40,
            eos_token_id=tokenizer.eos_token_id
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))

