
import os
from datasets import load_dataset
from typing import List, Tuple

# Import your evaluation function
# (Paste or import evaluate_image_qa from your "evaluation.py". 
#  We'll assume you have it in the same folder for demonstration.)
from evaluation_local import evaluate_image_qa

def load_freedomintelligence_dataset(
    split: str = "train",
    sample_size: int = 5
) -> List[Tuple[str, str, str]]:
    """
    Loads 'FreedomIntelligence/Medical_Multimodal_Evaluation_Data' from Hugging Face,
    converts to a list of (question, image_path, answer) tuples.

    :param split: "train", "validation", or "test"
    :param sample_size: how many examples to load for demonstration
    :return: A list of (user_question, image_filepath, reference_answer)
    """
    ds = load_dataset("FreedomIntelligence/Medical_Multimodal_Evaluation_Data", split="test")


    # Each row in ds has columns: 
    #   question (str), image (list[str]), options (list[str]), answer (str), etc.
    # The 'image' key typically is something like ["images/ankle071718.png"].

    results = []
    for i in range(min(sample_size, len(ds))):
        example = ds[i]

        # 1) Extract question
        question = example["question"]

        # 2) Extract answer
        reference_answer = example["answer"]

        # 3) Extract the image file path
        #    By default, 'image' is a list containing one or more paths, e.g. ["images/ankle071718.png"]
        #    We'll just take the first element if there's exactly one.
        if not example["image"]:
            # If there's no image path, skip or handle differently
            continue

        image_path = example["image"][0]  # e.g. "images/ankle071718.png"

        # Now you must ensure that "images/ankle071718.png" actually exists on disk. 
        # If the dataset is local and the images folder is at the same level, then 
        # "image_path" must be accessible for your 'encode_image' function to read it.

        # Collect
        results.append((question, image_path, reference_answer))

    return results


if __name__ == "__main__":
    # 1) Load the dataset (train split for demonstration)
    data_for_eval = load_freedomintelligence_dataset(split="train", sample_size=5)

    # 'data_for_eval' is now a list of (user_text, image_path, ref_answer)
    # Example: 
    # [
    #   ("What can be observed in this image?", "images/ankle071718.png", "Plantar fascia pathology"),
    #   ...
    # ]

    # 2) Call your evaluate_image_qa function from "evaluation.py"
    #    We'll assume you want "fra" as target language for translation.
    from evaluation_local import evaluate_image_qa  # if not already imported above

    evaluate_image_qa(data_for_eval, target_language="fra")
