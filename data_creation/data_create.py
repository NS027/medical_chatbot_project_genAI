from datasets import Dataset, DatasetDict, Features, Value, Image
import json
import os

# Function to load jsonl and create dataset
def load_split(jsonl_path, image_root):
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            image_path = os.path.join(image_root, os.path.basename(item['image_path']))
            data.append({
                "image": image_path,
                "question": item["question"],
                "answer": item["answer"]
            })
    return Dataset.from_list(data, features=Features({
        "image": Image(),
        "question": Value("string"),
        "answer": Value("string"),
    }))

# Paths
dataset_repo = "SiyunHE/medical-pilagemma-lora" 

train_dataset = load_split("dataset_train/medical_qa_lora.jsonl", "dataset_train/images")
test_dataset  = load_split("dataset_test/medical_qa_lora.jsonl", "dataset_test/images")

dataset_dict = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

# Push to Hugging Face Hub
dataset_dict.push_to_hub(dataset_repo, private=True)

print(f"Dataset uploaded to: https://huggingface.co/datasets/{dataset_repo}")

# Upload README.md
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="README.md",
    path_in_repo="README.md",
    repo_id="SiyunHE/medical-pilagemma-lora",
    repo_type="dataset"
)

# Make the dataset public
from huggingface_hub import HfApi

api = HfApi()
api.update_repo_visibility(
    repo_id="SiyunHE/medical-pilagemma-lora",
    repo_type="dataset",
    private=False
)