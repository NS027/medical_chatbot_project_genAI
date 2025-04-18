# 🦷 Two-Stage Fine-Tuning: AI Dental Doctor (Mistral-7B-Instruct + LoRA)

This project performs a two-stage LoRA fine-tuning using [Unsloth](https://github.com/unslothai/unsloth) on `Mistral-7B-Instruct`, focusing on building a friendly and domain-specific AI doctor that specializes in **dental implantation**.

---

## ✅ Models to Compare

1. `mistralai/Mistral-7B-Instruct-v0.1` (Base model)
2. `ritvik77/Medical_Doctor_AI_LoRA-Mistral-7B-Instruct_FullModel` (Pretrained LoRA)
3. `My fine-tuned LoRA` (from Stage 1 + 2)

---

## ✅ Workflow Overview

```
[Stage 1: Behavior Tone Training]
         ↓
[LoRA Adapter 1 Saved]
         ↓
[Stage 2: Implant Domain Fine-Tuning]
         ↓
[LoRA Adapter 2: Your Final AI Model]
         ↓
[Compare all 3 models: base, LoRA-found, your LoRA]
```

---

## 📦 Directory Structure

```
.
├── data/
│   ├── stage1_med_dialog.jsonl           # Tone dataset (MedDialog-EN)
│   └── stage2_implantation_data.jsonl    # Dental implantation Q&A
├── outputs/
│   ├── stage1_lora_adapter/
│   └── stage2_lora_adapter/
├── notebooks/
│   ├── stage1_train_behavior.ipynb
│   └── stage2_train_domain.ipynb
├── compare_models.py
└── README.md
```

---

## 🚀 Stage 1: Behavior Fine-Tuning (Empathy, Tone)

### 📌 Dataset: [`UCSD26/medical_dialog` ](https://huggingface.co/datasets/UCSD26/medical_dialog)   
Convert to JSONL format:
```json
{
  "prompt": "### Instruction:\nHow to treat sore throat at home?\n\n### Response:",
  "response": "To treat a sore throat at home, try warm salt water gargling..."
}
```

### 🔧 Training
- Use Unsloth + LoRA (4-bit)
- 1–2 epochs is enough
- Save to: `outputs/stage1_lora_adapter/`

---

## 🧠 Stage 2: Implantation Domain Fine-Tuning

### 📌 Dataset: Your custom dental Q&A  
Example:
```json
{
  "prompt": "### Instruction:\nWhat is guided bone regeneration before implant?\n\n### Response:",
  "response": "Guided bone regeneration (GBR) is a procedure that uses barrier membranes to encourage bone regrowth prior to implant placement..."
}
```

### 🔧 Training
- Load `Mistral-7B-Instruct` + Stage 1 adapter
- Train 1–3 epochs on `stage2_implantation_data.jsonl`
- Save to: `outputs/stage2_lora_adapter/`

---

## 📊 Evaluation Setup

Create a prompt list:
```
1. What are the steps in a sinus lift procedure?
2. Can diabetics receive dental implants safely?
3. How long does osseointegration take?
...
```

Compare responses from:
- `mistral` (Ollama CLI or Colab)
- `ritvik77/...` (Colab)
- `your model` (load adapter in Colab)

Evaluate:
| Prompt | Base | Pretrained | Your LoRA |
|--------|------|------------|-----------|
| Q1     | 3    | 4          | **5**     |
| Q2     | 2    | 5          | **5**     |

---

## ✅ Summary

| Step         | Output                          |
|--------------|----------------------------------|
| Stage 1 LoRA | `stage1_lora_adapter/`          |
| Stage 2 LoRA | `stage2_lora_adapter/`          |
| Final model  | Domain-aware, polite, structured dental AI doctor |
| Comparison   | Clear difference in tone + terminology vs base/pretrained |

---

## 🛠 Optional: Next Steps

- Export your LoRA adapter and merge with base for deployment
- Load model into AnythingLLM or Ollama
- Fine-tune further with more user-generated data