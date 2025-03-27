import os
import streamlit as st
import whisper
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

# Hugging Face model details
REPO_ID = "bhavanisankar-45/mistral"  # Replace with your Hugging Face repo
MODEL_FILENAME = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# Fetch model from Hugging Face
st.write("Fetching AI model from Hugging Face... This may take a while.")
MODEL_PATH = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
st.write("Model loaded successfully!")

# Load LLaMA model
llm = Llama(model_path=MODEL_PATH, n_ctx=4096)

def transcribe_audio(audio_path):
    """Transcribe audio using Whisper."""
    model = whisper.load_model("base")
    try:
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return "Error processing audio."

def generate_soap_note(conversation):
    """Generate a structured SOAP note."""
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
    output = llm(prompt=prompt, max_tokens=400, temperature=0.3, stop=["###"])
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
