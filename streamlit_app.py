import os
import streamlit as st
import tempfile
import torchaudio
import whisper
from llama_cpp import Llama
from docx import Document
from huggingface_hub import hf_hub_download

# Hugging Face model details
REPO_ID = "bhavanisankar-45/mistral"  # Replace with your Hugging Face repo
MODEL_FILENAME = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# Ensure 'models' directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Download model from Hugging Face if not present
if not os.path.exists(MODEL_PATH):
    st.write("Downloading model from Hugging Face... This may take a while.")
    hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME, local_dir=MODEL_DIR)
    st.write("Download complete!")
else:
    st.write("Model found locally. Skipping download.")

# Load LLaMA model with 1024 tokens
st.write("Loading AI model...")
llm = Llama(model_path=MODEL_PATH, n_ctx=1024)
st.write("Model loaded successfully!")

def transcribe_audio(audio_path):
    """Load audio using torchaudio and transcribe with Whisper."""
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
    """Generate a structured SOAP note from the transcription using the LLaMA model."""
    prompt = f"""
    Convert the following conversation into a structured SOAP note.

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

    Conversation:
    {conversation}

    SOAP Note:
    """

    # Generate response from LLaMA model
    output = llm(
        prompt=prompt,
        max_tokens=400,
        temperature=0.3,
        stop=["###"]
    )

    return output["choices"][0]["text"].strip()

def save_soap_note_to_word(soap_note):
    """Save the SOAP note as a properly formatted Word document and return the file path."""
    doc = Document()
    doc.add_heading("SOAP Note", level=1)

    sections = soap_note.split("\n\n")  # Split based on double newlines
    for section in sections:
        if ":" in section:  # Ensures it's a section header
            lines = section.split("\n")
            doc.add_heading(lines[0].strip(), level=2)  # Bold the section title
            for line in lines[1:]:  # Add bullet points for each line
                if ":" in line:
                    doc.add_paragraph(f"• {line.strip()}")  # Bullet point
                else:
                    doc.add_paragraph(line.strip())  # Regular text

    temp_word_file = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
    doc.save(temp_word_file.name)
    return temp_word_file.name

# Streamlit UI
st.title("AI-Based Medical Documentation Assistant")

option = st.radio("Choose an option:", ("Upload Audio",))
audio_file = None
transcription = None
soap_note = None

if option == "Upload Audio":
    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            audio_file = temp_file.name

        st.audio(audio_file, format="audio/wav")

        # Analyze button
        if st.button("Analyze"):
            st.write("Transcribing...")
            transcription = transcribe_audio(audio_file)
            st.subheader("Transcription:")
            st.write(transcription)

            # Generate SOAP note
            st.write("Generating SOAP Note...")
            soap_note = generate_soap_note(transcription)
            st.subheader("SOAP Note:")
            st.write(soap_note)

            # Save and provide download option
            soap_note_file = save_soap_note_to_word(soap_note)
            with open(soap_note_file, "rb") as file:
                st.download_button(
                    label="Download SOAP Note",
                    data=file,
                    file_name="SOAP_Note.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )

        # Cleanup after analysis
        if audio_file:
            os.remove(audio_file)
