# if you dont use pipenv uncomment the following:
# from dotenv import load_dotenv
# load_dotenv()

# VoiceBot UI with Gradio
import os
import gradio as gr

from brain_of_the_doctor import encode_image, analyze_image_with_query, summarize_doctor_response, translate_response, analyze_with_gemma
from voice_of_the_patient import transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_gtts

system_prompt = """
Act as a professional medical doctor. Analyze the uploaded image carefully.

If there is anything medically concerning, describe it in simple, human-friendly language. If applicable, suggest possible conditions (differential diagnosis) and potential remedies.

Guidelines for your response:
- Do not include any numbers, symbols, or special characters.
- Respond in a single, natural paragraph (maximum 2 sentences).
- Never say phrases like 'In the image I see'. Instead, start directly with statements like 'With what I see, I think you have...'
- Never mention you are an AI or language model.
- Do not use markdown formatting.
- Write as if speaking directly to a patient in a real consultation.
- Be clear, concise, and warm.
"""

def process_inputs(audio_filepath, image_filepath, target_language, user_text=None, model_choice="llama"):
    if user_text:
        user_input = user_text
    elif audio_filepath:
        user_input = transcribe_with_groq(
            GROQ_API_KEY=os.environ.get("GROQ_API_KEY"),
            audio_filepath=audio_filepath,
            stt_model="whisper-large-v3"
        )
    else:
        return "No input provided", "No response", "No summary", "No translation"

    if not image_filepath:
        doctor_response = "No image provided for me to analyze"
    else:
        # Different model choices
        if model_choice == "llama (advance)":
            doctor_response = analyze_image_with_query(
                query=system_prompt + user_input,
                encoded_image=encode_image(image_filepath),  # Base64 for Groq
                #model="meta-llama/llama-3.2-11b-vision-instruct:free"
            )
        elif model_choice == "gemma (basic)":
            doctor_response = analyze_with_gemma(
                query=user_input,
                image_path=image_filepath,  
            )
        else:
            doctor_response = "Invalid model choice."

    # Summarize the response
    summary = summarize_doctor_response(doctor_response)

    # Translate the doctor's response
    translated_response = translate_response(doctor_response, target_language)

    # limited_doctor_response = " ".join(doctor_response.split()[:30])
    # Generate voice of the translated response
    voice_of_doctor = text_to_speech_with_gtts( 
        input_text= doctor_response,
        output_filepath="audio/final.mp3"
    )

    return user_input, doctor_response, summary, translated_response

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft(primary_hue="purple", secondary_hue="blue")) as demo:
    gr.Markdown("<h1 style='text-align: center; font-weight: bold;'>🧑‍⚕️ AI Doctor: Vision and Voice-Powered Diagnosis 🧑‍⚕️</h1>")

    with gr.Column():
        with gr.Row():
            gr.Markdown(
                "<p style='text-align: center; font-size: 18px;'>🎤 Use your microphone or type your symptoms directly. "
                "You can also upload a medical image for visual diagnosis.</p>"
            )

        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Record your Symptoms")
                text_input = gr.Textbox(label="Or type your symptoms here")
                speech_to_text_output = gr.Textbox(label="Processed Input (Text)")

            image_input = gr.Image(type="filepath", label="Upload Medical Image")

        with gr.Row():
            doctor_response = gr.Textbox(label="Doctor's Response")
            summary = gr.Textbox(label="Summary")

        with gr.Row():
            translated_response = gr.Textbox(label="Translated Response")
            
            language_dropdown = gr.Dropdown(
                choices=["ita", "spa", "fra"],
                value="fra",
                label="Select translation language"
            )
            model_dropdown = gr.Dropdown(
                    choices=["llama (advance)", "gemma (basic)"],
                    value="llama (advance)",
                    label="Choose Model"
                )
            
        gr.Markdown(
            "<p style='color: purple; text-align: center;'>⚠️ Please select both a model and a target language before clicking Diagnose.</p>"
        )

        process_button = gr.Button("Diagnose")

        gr.Markdown(
            """
            <details style="margin-top: 15px; font-size: 15px;">
            <summary>⚠️ <strong>Important Disclaimer — Please Read</strong></summary>
            <div style="margin-top: 10px; color: #b71c1c;">
                This AI Doctor is <strong>not a licensed medical professional</strong> and should <strong>not be used as a substitute</strong> for real medical advice, diagnosis, or treatment.<br><br>
                For any health concerns, symptoms, or medical decisions, you should <strong>always consult a qualified healthcare provider</strong>.<br><br>
                Uploaded images are processed by machine learning models and <strong>not stored</strong>, but please be aware that you are uploading at your own risk. <strong>Do not submit personal, sensitive, or identifying medical images.</strong>
            </div>
            </details>
            """
        )

        process_button.click(
        fn=process_inputs,
        inputs=[audio_input, image_input, language_dropdown, text_input, model_dropdown],
        outputs=[speech_to_text_output, doctor_response, summary, translated_response]
    )

if __name__ == "__main__":
    demo.launch(debug=True)