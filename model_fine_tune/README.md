# 🦷 Dental Code Fine-Tuning – Qwen2.5-1.5B (LoRA)

This folder contains the implementation and results of fine-tuning the `Qwen/Qwen2.5-1.5B` large language model on a dental domain dataset using Low-Rank Adaptation (LoRA). The goal was to adapt a lightweight, instruction-following LLM to generate detailed, patient-friendly explanations of ADA dental procedure codes. This component serves as a subsidy module within the broader **AI Doctor** multimodal medical chatbot system.

## 📁 Folder Structure



---

## 🧠 Model Overview

- **Base Model**: `Qwen/Qwen2.5-1.5B`
- **Fine-Tuned Adapter**: `BirdieByte1024/Qwen2.5-1.5B-LoRA-dental`
- **Full Merged Model**: `BirdieByte1024/Qwen2.5-1.5B-dental-full`

> All models are hosted publicly on the Hugging Face Hub.

---

## 🧪 Dataset

- **Source**: `TachyHealth/ADA_Dental_Code_to_SBS_V2` (Hugging Face)
- Contains ADA codes, short descriptions, and detailed procedure explanations.
- Preprocessing: filtered nulls, reformatted into instruction-style prompts

Example prompt format:
```
### Instruction:
Given an ADA dental procedure code and its short name, provide a detailed explanation that helps a patient understand what the procedure involves.

### Code:
39 - CT of facial bone

### Response:
<Expected explanation>
```

---

## ⚙️ Training Details

- **Framework**: Hugging Face `transformers`, `peft`, `datasets`
- **Hardware**: NVIDIA GTX 1060 (6GB VRAM)
- **Batch Size**: 1
- **Epochs**: 1
- **Precision**: FP32
- **LoRA Config**:
  - `r=8`, `lora_alpha=16`
  - Target modules: `q_proj`, `v_proj`, `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

---

## 📊 Results

The fine-tuned model learned the domain-specific structure and terminology well, generating readable and medically aligned explanations. However, minor repetition artifacts were observed, likely due to overfitting and low batch size.

Sample outputs are included in `output/screenshots.png` and referenced in the main project report.

---

## 🩺 Role in AI Doctor

This model acts as a lightweight, locally deployable reasoning module inside the AI Doctor system. It is ideal for:
- Quick offline lookups of dental codes
- Voice-assisted responses to code-based queries
- Reducing load on the main 11B reasoning model

---

## 🔮 Future Work

- Apply repetition penalty and early stopping
- Fine-tune with a more diverse prompt pool
- Add multilingual support (e.g., French, Chinese)
- Quantize and convert to GGUF for use with Ollama

---

## 📎 Contributors

- Project: AI Doctor – Northeastern University Capstone  
- Term: Spring 2025

---
