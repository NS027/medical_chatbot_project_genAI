# Additional imports for NLP
import spacy

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")

# Existing imports
from dotenv import load_dotenv
load_dotenv()

# Step1: Setup GROQ API key
import os

# Access API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if GROQ_API_KEY is None:
    raise ValueError("GROQ_API_KEY is not set.")

# Step2: Convert image to required format
import base64

def encode_image(image_path):
    image_file=open(image_path, "rb")
    return base64.b64encode(image_file.read()).decode('utf-8')

# Step3: Setup Multimodal LLM
from groq import Groq

query="Is there something wrong with my face?"
model="llama-3.2-90b-vision-preview"

def analyze_image_with_query(query, model, encoded_image):
    client=Groq()
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": query
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                    },
                },
            ],
        }]
    chat_completion=client.chat.completions.create(
        messages=messages,
        model=model
    )

    return chat_completion.choices[0].message.content

from transformers import pipeline
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Initialize the summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_doctor_response(response_text, max_length=50):
    """Summarize the response in real-time."""
    summary = summarizer(response_text, max_length=max_length, min_length=20, do_sample=False)
    return summary[0]["summary_text"]

# Translation
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

    # Print supported language codes for reference
    # You can remove or comment out this line in production
    print("Supported Language Codes:", tokenizer.supported_language_codes)

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