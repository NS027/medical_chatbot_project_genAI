# Additional imports for NLP
import spacy
import os
import json
import base64
import requests
from dotenv import load_dotenv

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")

# Step1: Setup GROQ API key
# import os

# # Access API Key
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# if GROQ_API_KEY is None:
#     raise ValueError("GROQ_API_KEY is not set.")

# Step1: Setup OpenRouter API key
load_dotenv()  

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if OPENROUTER_API_KEY is None:
    raise ValueError("OPENROUTER_API_KEY is not set.")

# Step2: Convert image to required format
def encode_image(image_path):
    image_file=open(image_path, "rb")
    return base64.b64encode(image_file.read()).decode('utf-8')

### We use Groq API to call LLaMA 3.2 90B Vision-Instruct model. But on April 16, 2025, we found that the model is no longer supported by Groq API.###
# # Step3: Setup Multimodal LLM
# from groq import Groq

# query="Is there something wrong with my face?"
# model="llama-3.2-90b-vision-preview"

# def analyze_image_with_query(query, model, encoded_image):
#     client=Groq()
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": query
#                 },
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": f"data:image/jpeg;base64,{encoded_image}",
#                     },
#                 },
#             ],
#         }]
#     chat_completion=client.chat.completions.create(
#         messages=messages,
#         model=model
#     )

#     return chat_completion.choices[0].message.content

# Step 3: Use OpenRouter API to analyze image + text
def analyze_image_with_query(query, encoded_image):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    # Convert image to data URI format
    image_data_uri = f"data:image/jpeg;base64,{encoded_image}"

    payload = {
        "model": "meta-llama/llama-3.2-90b-vision-instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image_url", "image_url": {"url": image_data_uri}}
                ]
            }
        ]
    }

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        data=json.dumps(payload)
    )

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"OpenRouter request failed: {response.status_code} - {response.text}")



from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from huggingface_hub import login
import os
from PIL import Image
import torch

# # Load .env (optional if you use dotenv)
# from dotenv import load_dotenv
# load_dotenv()

# Hugging Face Auth
login(token=os.getenv("HF_TOKEN"))

from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel, PeftConfig
from PIL import Image
import torch

# Load your LoRA model from HuggingFace
model_id = "SiyunHE/medical-pilagemma-lora"
base_model_id = "google/paligemma-3b-pt-224"

# Load processor
processor = AutoProcessor.from_pretrained(base_model_id, use_fast=True)

# Load base model config from LoRA
peft_config = PeftConfig.from_pretrained(model_id)

# Load base model
base_model = PaliGemmaForConditionalGeneration.from_pretrained(base_model_id)

# Load LoRA weights
model = PeftModel.from_pretrained(base_model, model_id).eval()

# Inference function
def analyze_with_gemma(query, image_path):
    image = Image.open(image_path).convert("RGB")
    query = query.strip()
    prompt = "<image> " + query

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False
        )
        # Only keep new generated tokens
        output = outputs[0][input_len:]

        response = processor.decode(output, skip_special_tokens=True)

    return response


# Step4: Summarize the response
from transformers import pipeline
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Initialize the summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_doctor_response(response_text):
    """Summarize the response in real-time."""
    # if reponse_text is less than 20 words, return it as is
    if len(response_text.split()) < 20:
        return response_text
    # calculate max_length to be 50% of the response length
    max_length = int(len(response_text.split()) * 0.2)
    # calculate min_length to be 20% of the response length
    min_length = int(len(response_text.split()) * 0.1) 
    summary = summarizer(response_text, max_length, min_length, do_sample=False)
    return summary[0]["summary_text"]

# Step5: Translate the response
from transformers import MarianMTModel, MarianTokenizer

def translate_response(response_text, target_language):
    """
    Translates the input text to the target language using the MarianMTModel.

    :param response_text: Text to be translated.
    :param target_language: Target language code (use supported codes by the model).
    :return: Translated text.
    """

    # Define the model name
    model_name = "Helsinki-NLP/opus-mt-en-roa"

    # Load the tokenizer and model
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Check if the target language is supported
    if f'>>{target_language}<<' not in tokenizer.supported_language_codes:
        raise ValueError(f"Unsupported language code: {target_language}")

    # Prepare source text with target language token
    src_text = f'>>{target_language}<< {response_text}'

    # Tokenize input text
    inputs = tokenizer([src_text], return_tensors="pt", padding=True, truncation=True)

    # Generate translation
    translated_tokens = model.generate(**inputs)

    # Decode translated text
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

    return translated_text