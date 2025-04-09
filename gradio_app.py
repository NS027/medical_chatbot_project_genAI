# if you dont use pipenv uncomment the following:
# from dotenv import load_dotenv
# load_dotenv()

# VoiceBot UI with Gradio
import os
import gradio as gr

from brain_of_the_doctor import encode_image, analyze_image_with_query, summarize_doctor_response, translate_response
from voice_of_the_patient import transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_gtts

system_prompt="""You have to act as a professional doctor, i know you are not but this is for learning purpose.
            What's in this image?. Do you find anything wrong with it medically?
            If you make a differential, suggest some remedies for them. Donot add any numbers or special characters in
            your response. Your response should be in one long paragraph. Also always answer as if you are answering to a real person.
            Donot say 'In the image I see' but say 'With what I see, I think you have ....'
            Dont respond as an AI model in markdown, your answer should mimic that of an actual doctor not an AI bot,
            Keep your answer concise (max 2 sentences). No preamble, start your answer right away please"""

def process_inputs(audio_filepath, image_filepath, target_language, user_text=None):
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

    if image_filepath:
        doctor_response = analyze_image_with_query(
            query=system_prompt + user_input,
            encoded_image=encode_image(image_filepath),
            model="llama-3.2-11b-vision-preview"
        )
    else:
        doctor_response = "No image provided for me to analyze"

    # Summarize the response
    summary = summarize_doctor_response(doctor_response)

    # Translate the doctor's response
    translated_response = translate_response(doctor_response, target_language)

    # limited_doctor_response = " ".join(doctor_response.split()[:30])
    # Generate voice of the translated response
    voice_of_doctor = text_to_speech_with_gtts( 
        input_text= doctor_response,
        output_filepath="final.mp3"
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

        process_button = gr.Button("Diagnose")

        process_button.click(
            fn=process_inputs,
            inputs=[audio_input, image_input, language_dropdown, text_input],
            outputs=[speech_to_text_output, doctor_response, summary, translated_response]
        )

if __name__ == "__main__":
    demo.launch(debug=True)