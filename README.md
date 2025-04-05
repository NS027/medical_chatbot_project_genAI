# Medical ChatBot with Multimodal LLM

## note from Siyun:
### progress 1:
1. create a new envionment (use python 3.11) first and run the requirements.txt to install all package.
2. We need to register the API key for GROQ_API_KEY and ELEVENLABS_API_KEY to run the code. These API keys are free.
3. only .py code files are related to our project. Others are just some output audio files.

This the original Github repo from the youtube:
https://github.com/AIwithhassan/ai-doctor-2.0-voice-and-vision

We’ve removed redundant code and files from our repo. We can start with this bare minimum and build on it. <br>

### progress 2:
Add a functionality: summary. Summerize the doctor's reponse using Spacy model. Update the gradio interface accordingly. Experiment with two method:
1. Spacy: This approach extracts key sentences based on importance and length. (Simple and keeps only the first two sentences as a summary.)
2. BART: LLM that has stronger ability ( high-quality, abstractive summary that paraphrases and condenses the content)

### progress 3:
Add translation functionality. Experiment two method. Translator from googletrans package and MarianMTModel from transformer:
Googletrans supports more languages but the newest version only Coroutine. Cannot output text directly. The older version can output text directly, but if using the old version will cause version conflict with other packages in our dev environment.

### progess 4:
updata UI. Make it prettier(desing, colors, and font) and more user-friendly. 

## Overview
This project aims to build a **Medical ChatBot** using a **Multimodal Large Language Model (LLM)**. The chatbot will be capable of processing text and voice inputs from patients and generating text and voice responses as a virtual doctor. 

## Features
- **Multimodal LLM**: Processes both text and image inputs
- **Speech-to-Text (STT)**: Converts patient speech to text using transcription models
- **Text-to-Speech (TTS)**: Converts generated text responses to voice
- **Interactive UI**: A user-friendly interface built with Gradio for interaction

## Project Layout
### Phase 1 – Setup the Brain of the Doctor (Multimodal LLM)
- Setup **GROQ API key**
- Convert images to required formats
- Setup and integrate **Multimodal LLM**

### Phase 2 – Setup Voice of the Patient
- Configure **Audio Recorder** using `ffmpeg` & `portaudio`
- Implement **Speech-to-Text (STT)** for transcription

### Phase 3 – Setup Voice of the Doctor
- Implement **Text-to-Speech (TTS)** using `gTTS` & `ElevenLabs`
- Generate voice responses from the chatbot’s text output

### Phase 4 – Setup UI for the VoiceBot
- Design an interactive **VoiceBot UI** using `Gradio`

## Installation
### Prerequisites
- Python 3.9+
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
python app.py
```

## Future Enhancements
- Support for multiple languages
- Improved speech synthesis with real-time response
- Integration with electronic health records (EHR)

## License
This project is open-source and available under the MIT License.

## Contributors
- **Contributors** – Open for contributions!

