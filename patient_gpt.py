import os
import streamlit as st
import tempfile
import torchaudio
import whisper
from llama_cpp import Llama

# Define model path
MODEL_PATH = os.path.join(os.getcwd(), "models", "mistral-7b-instruct-v0.2.Q4_K_M.gguf")

# Load LLaMA model
st.write("Loading AI model...")
llm = Llama(model_path=MODEL_PATH, n_ctx=4096)
st.write("Model loaded successfully!")

def transcribe_audio(audio_path):
    """Transcribe audio using Whisper."""
    model = whisper.load_model("base")
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        audio = waveform.numpy().flatten()
        result = model.transcribe(audio)
        return result["text"]
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return "Error processing audio."

def generate_soap_note(conversation):
    """Generate a SOAP note from user description."""
    prompt = f"""
    Convert the following description into a structured SOAP note.

    Patient Details:
    Name: Extract if mentioned, else "Not Mentioned"
    Age: Extract if mentioned, else "Not Mentioned"
    Gender: Extract if mentioned, else "Not Mentioned"

    Subjective:
    Main complaint and symptoms
    Medical history

    Objective:
    Physical exam findings or "Not Available"
    Test results or "Not Available"

    Assessment:
    Possible diagnosis
    Probable disease

    Plan:
    Medications, tests, and follow-up

    Description:
    {conversation}

    SOAP Note:
    """

    output = llm(
        prompt=prompt,
        max_tokens=400,
        temperature=0.3,
        stop=["###"]
    )

    return output["choices"][0]["text"].strip()

st.title("PatientGPT - Describe Your Symptoms")

input_type = st.radio("Choose input method:", ("Text", "Audio"))

if input_type == "Text":
    user_input = st.text_area("Describe your symptoms in detail")
    if st.button("Analyze"):
        if user_input.strip():
            st.write("Generating SOAP Note...")
            soap_note = generate_soap_note(user_input)
            st.subheader("SOAP Note:")
            st.write(soap_note)
        else:
            st.warning("Please enter some text.")
