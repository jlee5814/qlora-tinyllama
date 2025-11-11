from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer
import torch

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Load model in 4-bit QLoRA mode
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,
    device_map="auto"
)

model = prepare_model_for_kbit_training(model)

lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_cfg)

# dataset
dataset = load_dataset(
        "databricks/databricks-dolly-15k",
        split="train[:2000]"
)

def format(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
    }

# Build instruction/response text
dataset = dataset.map(
        format,
        remove_columns=dataset.column_names
)

# Tokenize into input_ids/attention_mask
def tokenize(example):
    output = tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    return output

dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"]
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./qlora-output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    warmup_steps=10,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=10,
    fp16=True,
    optim="paged_adamw_8bit",

    save_strategy="epoch",
    save_steps=1000,
    group_by_length=False
)

print("FINAL dataset size:", len(dataset))

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

trainer.train()

