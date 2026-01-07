# QLoRA Fine-Tuning Demo (TinyLlama 1.1B)

## Overview
This project demonstrates **parameter-efficient fine-tuning under strict GPU memory constraints** using QLoRA on the TinyLlama 1.1B model.

The goal is not peak model quality, but to explore **what training configurations are feasible on consumer-grade hardware**, and what tradeoffs are introduced when memory becomes the dominant constraint.

All experiments were run on a **single NVIDIA RTX 3070 (8GB VRAM)**.

---

## System Context

Fine-tuning modern language models is often bottlenecked by GPU memory rather than compute.  
QLoRA addresses this by combining:

- **4-bit quantization** to reduce base model memory footprint
- **LoRA adapters** to limit the number of trainable parameters
- **PEFT** to decouple fine-tuning from full model weights

This setup allows billion-parameter models to be adapted on hardware that would otherwise be insufficient.

---

## Configuration Summary

- **Base model:** TinyLlama 1.1B  
- **Dataset:** Dolly-15K  
- **Fine-tuning method:** QLoRA (4-bit)  
- **Trainable parameters:** LoRA adapters only  
- **Hardware:** Single NVIDIA RTX 3070  
- **Frameworks:** PyTorch, Hugging Face Transformers, PEFT, bitsandbytes  

---

## Key Tradeoffs Observed

- **Memory vs stability:**  
  4-bit quantization significantly reduces VRAM usage, but increases sensitivity to learning rate and optimizer settings.

- **Throughput vs accessibility:**  
  Single-GPU training enables accessibility but limits batch size and slows convergence.

- **Adapter indirection:**  
  LoRA reduces training cost but introduces additional indirection at inference time.

These tradeoffs mirror real-world constraints faced in resource-limited or edge deployment environments.

---

## Repository Structure

- `train.py`  
  QLoRA fine-tuning script for TinyLlama.

- `inference.py`  
  Loads the base model + LoRA adapters and runs inference.

- `requirements.txt`  
  Python dependencies.

- `qlora-output/`  
  Saved LoRA adapter checkpoint (for inspection and experimentation).

---

## How to Train

```bash
python train.py
```

This assumes a CUDA-enabled environment with compatible NVIDIA drivers.

---

## Limitations

- Not optimized for multi-GPU or distributed training
- No throughput or latency benchmarking
- Hyperparameters tuned for feasibility, not optimal convergence
- Checkpoint is for experimentation, not production deployment

---

## Scope Clarification

This project is **not**:
- A production training pipeline
- A benchmark study
- A scalable fine-tuning framework

It **is**:
- A constrained-systems experiment
- A reference for memory-efficient fine-tuning
- A practical exploration of QLoRA tradeoffs on limited hardware
