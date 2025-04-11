# pip install openai==0.28 (newest version does not work because openai.ChatCompletion no longer exists)
import openai
import json
import os
from tqdm import tqdm

# Your OpenAI API Key here directly
openai.api_key = "PUT-API-KEY-HERE"

image_folder = "images"
output_file = "medical_qa_lora.jsonl"

image_list = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

output = []

for image_name in tqdm(image_list):
    image_path = os.path.join(image_folder, image_name)
    
    # Step 1: Generate Patient Question
    question_prompt = f"""
You are a patient noticing a medical problem based on the following image file name: {image_name}.
Generate a natural, short question a real patient would ask a doctor about this image.
Do not mention the word 'image' or 'file name'.
Write it like a real patient seeking help.
"""

    question_response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": question_prompt}],
        max_tokens=100,
    )
    question = question_response['choices'][0]['message']['content'].strip()

    # Step 2: Generate Doctor Answer
    answer_prompt = f"""
Act as a professional medical doctor specializing in dermatology and oral health.

A patient asked this question: "{question}"

Provide a natural, clear, warm, short answer covering:
- What might be the medical condition.
- Possible causes.
- Simple treatment advice.

Rules:
- Respond like a real doctor talking to a patient.
- No symbols, no markdown, no lists.
- about 2 sentences.
"""

    answer_response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": answer_prompt}],
        max_tokens=300,
    )
    answer = answer_response['choices'][0]['message']['content'].strip()

    # Append result
    output.append({
        "image_path": image_path,  # keep relative path
        "question": question,
        "answer": answer
    })

# Save to JSONL
with open(output_file, 'w', encoding='utf-8') as f:
    for item in output:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print("Done! Output saved to:", output_file)