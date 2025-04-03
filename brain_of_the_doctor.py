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

# def summarize_doctor_response(response_text, num_sentences=2):
#     """Summarizes the response by selecting the most relevant sentences."""
#     doc = nlp(response_text)
#     sentences = [sent.text.strip() for sent in doc.sents]

#     # Return the first 'num_sentences' sentences as a simple summary
#     summary = " ".join(sentences[:num_sentences]) if sentences else "No summary available."
    
#     return summary

# Summerize the doctor response in real-time
from transformers import pipeline

# Initialize the summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_doctor_response(response_text, max_length=50):
    """Summarize the response in real-time."""
    summary = summarizer(response_text, max_length=max_length, min_length=20, do_sample=False)
    return summary[0]["summary_text"]