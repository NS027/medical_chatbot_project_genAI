# Medical ChatBot with Multimodal LLM


### Overview

This project aims to develop a real-time **Medical ChatBot** powered by a **Multimodal Large Language Model (LLM)**. The chatbot will be able to handle both text and voice inputs from patients, and provide responses in text and speech. It will also include features such as summarization and translation, functioning as a virtual doctor.

Here shows our user interface:
![Alt text](UI.png)

## Features

- **Multimodal LLM**: Processes both text and image inputs
- **Speech-to-Text (STT)**: Converts patient speech to text using transcription models
- **Text-to-Speech (TTS)**: Converts generated text responses to voice
- **Interactive UI**: A user-friendly interface built with Gradio for interaction

## Project Layout

### Phase 1 – Setup the Brain of the Doctor (Multimodal LLM)

- Setup **GROQ API key**, **OpenRouter API Key**, **Hugging Face token** for 
paligemma-3b-pt-224 
- Convert images to required formats
- Setup and integrate **Multimodal LLM**

### Phase 2 – Setup Voice of the Patient

- Configure **Audio Recorder** using `ffmpeg` & `portaudio`
- Implement **Speech-to-Text (STT)** for transcription

### Phase 3 – Setup Voice of the Doctor

- Implement **Text-to-Speech (TTS)** using `gTTS` 
- Generate voice responses from the chatbot’s text output

### Phase 4 – Setup UI for the VoiceBot

- Design an interactive **VoiceBot UI** using `Gradio`

## Installation

### Prerequisites

- Python 3.11+
- `ffmpeg` and `portaudio`
- Required libraries (install using the following command):
  ```sh
  pip install openai gradio pydub ffmpeg-python speechrecognition gtts elevenlabs
  ```

### Setup Instructions

1. Clone this repository:
   ```sh
   git clone https://github.com/yourusername/medical_chat_bot.git
   cd medical_chat_bot
   ```
2. Set up the environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Set up **GROQ API Key**:
   ```sh
   export GROQ_API_KEY='your_api_key_here'
   ```

## Run Environment

### 1. If you just need to run the Python file

First, create a virtual environment with Python 3.11:

```bash
python3.11 -m venv myenv
```

#### For macOS/Linux:

Activate the virtual environment using:

```bash
source myenv/bin/activate
```

#### For Windows:

Activate the virtual environment using:

```bash
myenv\Scripts\activate
```

### The Requirements

To install the required packages, run:

```bash
pip install -r requirements.txt
```

### 2. If you need to run under a Conda environment

Navigate to your project directory:

```
cd path/to/your/file/
```

Create a Conda environment with Python 3.11:

```
conda create --prefix ./venv python=3.11
```

Activate the environment:

```
conda activate ./venv
```

---

### 📦 Install Requirements

Install the Python packages listed in the cleaned requirements file:

```
pip install -r requirements_conda.txt
```

---

### 🔊 Install PyAudio (using Conda)

To avoid build errors with PyAudio on macOS, install it via Conda:

```
conda install -c conda-forge pyaudio
```

---

### 🌐 Install spaCy Language Model

`en_core_web_sm` is not available on PyPI and must be installed using spaCy:

```
python -m spacy download en_core_web_sm
```

### .env File

Please refer to `.env.example` for setting up your environment variables.

## Running the ChatBot

To start the chatbot with the UI:

```sh
python gradio_app.py
```

## Future Enhancements

- Support for multiple languages
- Improved speech synthesis with real-time response
- Integration with electronic health records (EHR)

## License

This project is open-source and available under the MIT License.

## Contributors

- **Contributors** – Open for contributions!

## Reference:
https://github.com/AIwithhassan/ai-doctor-2.0-voice-and-vision
