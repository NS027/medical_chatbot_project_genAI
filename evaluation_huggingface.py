
import os
from datasets import load_dataset
from typing import List, Tuple

# Import the evaluation function
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

        if not example["image"]:
            # If there's no image path, skip or handle differently
            continue

        image_path = example["image"][0] 

        # Collect
        results.append((question, image_path, reference_answer))

    return results


if __name__ == "__main__":
    # 1) Load the dataset (train split for demonstration)
    data_for_eval = load_freedomintelligence_dataset(split="train", sample_size=5)

    # 'data_for_eval' is now a list of (user_text, image_path, ref_answer)

    # 2) Call  evaluate_image_qa function from "evaluation.py"
    from evaluation_local import evaluate_image_qa 

    evaluate_image_qa(data_for_eval, target_language="fra")
