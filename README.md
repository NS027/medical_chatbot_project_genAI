<p align="center">
  <img src="./model_fine_tune/assets/banner.jpg" alt="AI Doctor Banner" width="100%" />
</p>

<p align="center"><em>Healthcare guidance through multimodal generative AI.</em></p>

<p align="center">
  <img src="https://img.shields.io/badge/version-1.0.0-blue" />
  <img src="https://img.shields.io/badge/license-MIT-green" />
  <img src="https://img.shields.io/badge/python-3.12-yellow" />
  <img src="https://img.shields.io/badge/LoRA-enabled-ff69b4" />
</p>

---

**A powerful multimodal assistant for healthcare: process voice, text, and medical images to deliver personalized, context-aware medical advice in real time.**

---

> 🧠 Powered by:  
> - LLaMA 3.2 11B Vision-Instruct  
> - Whisper-large-v3 for speech  
> - PaliGemma & Qwen LoRA adapters for domain-specific logic  
> - Hugging Face & Google TTS

---

# 🩺 AI Doctor: Real-Time Medical Assistance with Multimodal Large Language Model
## 🧠 Overview

This project introduces a real-time **AI Doctor** chatbot powered by a **Multimodal Large Language Model (LLM)**. It supports both **voice** and **text** input, understands **medical images**, and provides responses via **speech** and **text**. Additional features include summarization and translation — offering a natural and accessible healthcare assistant.

---

## 🌟 Features

- 🔍 **Multimodal LLM**  
  Handles both text and image inputs for medical understanding  
  _↪️ Implemented in `brain_of_the_doctor.py`_

- 🗣️ **Speech-to-Text (STT)**  
  Converts patient voice input into text using Whisper  
  _↪️ Implemented in `voice_of_the_patient.py`_

- 🔈 **Text-to-Speech (TTS)**  
  Generates spoken responses from AI-generated answers  
  _↪️ Implemented in `voice_of_the_doctor.py`_

- 💬 **Interactive Gradio UI**  
  Provides a simple, real-time interface for user interaction  
  _↪️ Implemented in `gradio_app.py`_
---

## 🧪 Our Contributions

### 1. 🧑‍⚕️ AI Doctor Application
A conversational AI that helps users make informed decisions about seeking medical care — potentially reducing unnecessary visits and improving access.
  
<p align="center">
  <img src="UI.png" alt="User Interface" width="80%"/>
</p>

---

### 2. 🗂️ Custom Dataset Creation

Most existing medical datasets lack paired image-text data and focus on modalities like CT or PET scans — not suitable for real-world symptom images. To address this:

- We built a medical VQA dataset using **user-uploaded images** and **GPT-4o-generated** Q&A pairs simulating real patient inquiries and expert-level answers.
🤗 Dataset available on [Hugging Face](https://huggingface.co/datasets/SiyunHE/medical-pilagemma-lora)
📂 Data creation details: See [`data_creation/`](data_creation)

---

### 3. 🔧 Fine-Tuning with LoRA

We fine-tuned two domain-specific lightweight language models using **LoRA** to extend the capabilities of our main 11B multimodal backbone (LLaMA 3.2 Vision-Instruct). These LoRA adapters enable **fast, targeted medical reasoning on lower-resource devices**.


#### 🧠 PaliGemma LoRA – Medical VQA

We fine-tuned **PaliGemma** using LoRA on a custom medical image-question-answering dataset to build a lightweight alternative to our main vision-language model.

- Applied LoRA to **cross-attention layers** for multimodal alignment  
- Trained using Hugging Face Trainer on **Google Colab A100**  
- **Hyperparameters**: learning rate `5e-5`, batch size `4`, epochs `3`  
- Result: Outperforms base PaliGemma on domain-specific VQA tasks

🤗 [Model on Hugging Face](https://huggingface.co/SiyunHE/medical-pilagemma-lora)  
📁 Fine-tuning details: See [`experiments/`](experiments)


#### 🦷 Qwen2.5-1.5B LoRA – Dental Code Explanations

We also fine-tuned **Qwen2.5-1.5B**, a small open-source causal LLM, to specialize in **explaining ADA dental procedure codes** in a patient-friendly way.

- Used **Low-Rank Adaptation** for parameter-efficient training  
- Trained for 1 epoch on a consumer GPU (GTX 1060, 6GB)  
- Instruction-style prompting using real medical code descriptions  
- Output shows strong format compliance and domain-specific vocabulary

📁 Fine-tuning notebook: [`/model_fine_tune/Finetun_LoRA_Qwen_Dental.ipynb`](model_fine_tune/Finetun_LoRA_Qwen_Dental.ipynb)  
🤗 [LoRA Adapter](https://huggingface.co/BirdieByte1024/Qwen2.5-1.5B-LoRA-dental)  
🤗 [Merged Model](https://huggingface.co/BirdieByte1024/Qwen2.5-1.5B-dental-full)

---

### 4. 📊 Model Evaluation

We evaluated model performance using **BERTScore F1** on 30 samples from our dataset, comparing:

- 🔬 **LLaMA-3.2-11B-Vision-Instruct** (model we used)
- 🧬 **MMed-LLaMA 3** (trained on medical data)

<p align="center">
  <img src="evaluation/answer_similarity_comparison.png" alt="Evaluation Chart" width="80%">
</p>

📈 Results: Our model consistently achieves higher semantic alignment with ground truth answers, indicating stronger response quality for real-world medical VQA tasks.

📁 evaluation details: See [`evaluation/`](evaluation)

---

## ⚙️ Setup Instructions

### 🛠️ Prerequisites

- 🐍 **Python 3.11+**

- 🔑 **Environment Variables:**
 👉 See `.env.example` for required environment variables.

  - `GROQ_API_KEY` — _(Free)_  
    Required for **speech-to-text (STT)** using `whisper-large-v3`.

  - `HF_TOKEN` — _(Free)_  
    Needed to load the **Google PaliGemma** model: `google/paligemma-3b-pt-224`.

  - `OPENROUTER_API_KEY` — _(Paid or Free)_  
    Used to access **meta-llama/llama-3.2-11b-vision-instruct**  
    > We currently use the **paid version** for more stable performance,  
    > but you may switch to the free version:  
    > `meta-llama/llama-3.2-11b-vision-instruct:free`

### 🛠️Setup steps

1. Clone this repository:
   ```bash
   git clone medical_chatbot_project_genAI
   cd medical_chatbot_project_genAI
   ```
2. create a new environment with conda (recommend)
   ```bash
   conda create --name ai_doctor python=3.11
   ```
   create a new environment without conde
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On macOS/Linux
   .\venv\Scripts\activate     # On Windows
   ```
3. activate the enviornment
   ```bash
   conda activate ai_doctor
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. start the application
   ```bash
   gradio gradio_app.py
   ```



### 🚀 Future Development

- Add real-time medical knowledge via **Retrieval-Augmented Generation (RAG)** to overcome LLM knowledge cutoffs.
- Use **Medical Communication Protocols (MCP)** for better scalability and healthcare system integration.
- Expand **language support** and enhance **medical reasoning** capabilities.
- Conduct **clinical validation** to assess safety and real-world effectiveness.



## 📚Reference:
https://github.com/AIwithhassan/ai-doctor-2.0-voice-and-vision <br>
https://github.com/RyanWangZf/MedCLIP
