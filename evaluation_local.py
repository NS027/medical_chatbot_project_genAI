import os
from typing import List, Tuple

# 3rd-party for metrics
import evaluate
import jiwer  # for Word Error Rate (pip install jiwer)

from gradio_app import process_inputs
from voice_of_the_patient import transcribe_with_groq

# Initialize evaluators
rouge_evaluator = evaluate.load("rouge")
bertscore_evaluator = evaluate.load("bertscore")

# 1. Evaluate Image QA
def evaluate_image_qa(
    data: List[Tuple[str, str, str]],
    target_language: str = "fra"
):
    """
    data: list of (user_text, image_filepath, reference_answer)
      - user_text: the question or any text user would provide
      - image_filepath: path to the medical image
      - reference_answer: ground truth answer (string)

    We call `process_inputs(...)` from your app. That calls
    'analyze_image_with_query()' if an image is present.
    Then we compare the doctor's response to the reference_answer
    using ROUGE & BERTScore.
    """
    predictions = []
    references = []

    for user_text, image_path, ref_answer in data:
        # Because the system *requires* an image, we pass image_path != None
        # audio_filepath=None since we are not testing STT here
        # user_text is appended to the system_prompt internally
        # target_language could be "fra", "ita", or "spa"

        speech_text, doctor_resp, summary, translated_resp = process_inputs(
            audio_filepath=None,
            image_filepath=image_path,
            target_language=target_language,
            user_text=user_text
        )

        # The variable doctor_resp is the direct output from your model
        predictions.append(doctor_resp)
        references.append(ref_answer)

    print("\n===== IMAGE QA EVALUATION =====")
    # 1) ROUGE
    rouge_result = rouge_evaluator.compute(predictions=predictions, references=references)
    print("ROUGE:", rouge_result)

    # 2) BERTScore
    bert_result = bertscore_evaluator.compute(
        predictions=predictions,
        references=references,
        lang="en"  # or 'en-sci' for more domain-specific
    )
    # Print average BERTScore
    print("BERTScore (average):")
    print(f"  Precision: {sum(bert_result['precision']) / len(bert_result['precision']):.4f}")
    print(f"  Recall:    {sum(bert_result['recall']) / len(bert_result['recall']):.4f}")
    print(f"  F1:        {sum(bert_result['f1']) / len(bert_result['f1']):.4f}")


# 2. Evaluate Speech-to-Text (WER)

def evaluate_stt(
    data: List[Tuple[str, str]]
):
    """
    data: list of (audio_filepath, reference_transcription).

    We call 'transcribe_with_groq' from voice_of_the_patient.py
    to get the predicted transcription. Then compute WER using jiwer.
    """
    references = []
    predictions = []

    for audio_path, ref_text in data:
        pred_text = transcribe_with_groq(
            stt_model="whisper-large-v3",
            audio_filepath=audio_path,
            GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
        )
        references.append(ref_text)
        predictions.append(pred_text)

    # Use jiwer for Word Error Rate
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip()
    ])

    wer = jiwer.wer(
        references,
        predictions,
        truth_transform=transformation,
        hypothesis_transform=transformation
    )

    print("\n===== SPEECH-TO-TEXT EVALUATION =====")
    print(f"WER (Word Error Rate): {wer:.4f}")


# 3. Example usage if run this file directly
if __name__ == "__main__":
    # Example IMAGE QA test set
    # Replace 'image1.jpg' or 'image2.jpg' with real paths in environment,
    # and fill in references with short correct answers
    image_qa_data = [
        ("What's wrong with the skin", "skin_rash.jpg", "It is skin rash and maybe caused by allergic reaction, eczema or insect bits."),
    ]

    evaluate_image_qa(image_qa_data, target_language="fra")

    # Example STT test set
    # Replace 'audio1.mp3' etc. with real audio files containing known transcripts
    stt_data = [
        ("gtts_testing.mp3", "Hi, this is group 7's AI doctor for the final project."),
  
    ]

    evaluate_stt(stt_data)
