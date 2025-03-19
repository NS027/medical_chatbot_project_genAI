# Medical ChatBot with Multimodal LLM

## A quick note from Siyun:
1. create a new envionment (use python 3.11) first and run the requirements.txt to install all package.
2. We need to register the API key for GROQ_API_KEY and ELEVENLABS_API_KEY to run the code. These API keys are free.
3. only .py code files are related to our project. Others are just some output audio files.

This the original Github repo from the youtube:
https://github.com/AIwithhassan/ai-doctor-2.0-voice-and-vision

We’ve removed redundant code and files from our repo. We can start with this bare minimum and build on it.

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

### 2. If you need to run the Jupyter Notebook

Navigate to your project directory:

```bash
cd path/to/your/file/
```

Create a Conda environment with Python 3.11:

```bash
conda create --prefix ./venv python=3.11
```

#### To activate the environment:

```bash
conda activate ./venv
```

### The Requirements

To install the required packages, run:

```bash
pip install -r requirements.txt
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

