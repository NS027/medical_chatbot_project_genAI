import os
import shutil
import random
from tqdm import tqdm

# Input dataset root
input_root = "Multimodal Captioning Dataset"

# Output folders
train_folder = "dataset_train/images"
test_folder = "dataset_test/images"

os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

categories = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]

counter = 0  # For unique filenames

for category in tqdm(categories):
    category_path = os.path.join(input_root, category)
    images = [f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if len(images) == 0:
        continue

    # Randomly select one image for test
    test_image = random.choice(images)

    for img in images:
        src_path = os.path.join(category_path, img)
        ext = os.path.splitext(img)[-1]  # keep original extension
        new_name = f"{category.replace(' ', '_')}_{counter}{ext}"
        counter += 1

        if img == test_image:
            dst_path = os.path.join(test_folder, new_name)
        else:
            dst_path = os.path.join(train_folder, new_name)

        shutil.copy2(src_path, dst_path)

print("Dataset split completed!")
print(f"Train images saved to: {train_folder}")
print(f"Test images saved to: {test_folder}")