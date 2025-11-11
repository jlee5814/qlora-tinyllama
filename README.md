# QLoRA Fine-Tuning Demo (TinyLlama 1.1B)
This project shows how to fine-tune a 1.1B TinyLlama model, with the following:
- Dolly-15K dataset using QLoRA  
- 4-bit quantization 
- PEFT 

All on a single NVIDIA RTX 3070 GPU.

# Contents
- `train.py` – QLoRA training script
- `inference.py` – Load LoRA adapter and run inference
- `requirements.txt` – Python dependencies
- `qlora-output/` – Fine-tuned LoRA weights (checkpoint)

## How to Train
```bash
python train.py

