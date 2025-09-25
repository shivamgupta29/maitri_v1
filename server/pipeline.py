import os
import base64
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from moviepy import VideoFileClip

# --- Note: Replace these placeholder imports and functions ---
# with your actual model files when you integrate.
from models.ser_model.speech import load_model as load_speech_model, speech_predict
from models.fer_model.facial import load_model as load_video_model, video_to_base64_frames, video_predict
from models.rag_model.rag_model import AstroAssistant
# --- End of placeholder section ---


# --- Configuration ---
# Disable GPU for TensorFlow if you are running on a CPU-only environment
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# --- Define Paths and Load Models (once at startup) ---
# This ensures models are loaded into memory only one time.
BASE_DIR = Path(__file__).resolve().parent
speech_model_path = BASE_DIR / "models" / "ser_model" / "ser_model.keras"
video_model_path = BASE_DIR / "models" / "fer_model" / "model.h5"
rag_text_path = BASE_DIR / "models" / "rag_model" / "text.txt"

# Initialize models globally
try:
    print("[INFO] Loading models...")
    speech_model = load_speech_model(str(speech_model_path))
    video_model = load_video_model(str(video_model_path))
    assistant = AstroAssistant(str(rag_text_path), model_name="gemma3:1b")
    print("[INFO] All models loaded successfully.")
except Exception as e:
    print(f"[ERROR] Critical error loading models: {e}")
    speech_model = None
    video_model = None
    assistant = None

# --- Helper Functions for Parallel Processing ---

def _process_audio_task(video_path, model):
    """
    Internal function to extract audio, convert to base64, and run prediction.
    """
    temp_audio_path = None
    try:
        # Create a temporary file for the extracted audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
            temp_audio_path = temp_audio_file.name

        # Use moviepy to extract the audio track from the video file
        with VideoFileClip(video_path) as video_clip:
            # logger=None prevents moviepy from printing messages to the console
            video_clip.audio.write_audiofile(temp_audio_path, codec='pcm_s16le', logger=None)

        # Read the audio file's bytes and encode them into a base64 string
        with open(temp_audio_path, 'rb') as f:
            audio_bytes = f.read()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        # Run the speech emotion prediction
        return speech_predict(model, audio_base64)
    except Exception as e:
        print(f"[ERROR] Audio processing task failed: {e}")
        return "audio_error"
    finally:
        # Clean up the temporary audio file
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)


def _process_video_task(video_path, model):
    """
    Internal function to convert video to frames and run facial prediction.
    """
    try:
        frames = video_to_base64_frames(video_path)
        if not frames:
            return "no_frames_detected"
        return video_predict(model, frames)
    except Exception as e:
        print(f"[ERROR] Video processing task failed: {e}")
        return "video_error"

# --- Main Integration Function (to be called by main.py) ---

def process_multimodal_input(video_bytes: bytes, user_text: str):
    """
    Orchestrates the entire multimodal analysis pipeline.
    This is the main function your FastAPI server will call.
    """
    # Guard clause: Check if models were loaded correctly at startup
    if not all([speech_model, video_model, assistant]):
        return "Error: One or more AI models failed to load. Please check server logs."

    # Use a temporary file to safely handle the uploaded video bytes
    with tempfile.NamedTemporaryFile(delete=True, suffix=".webm") as temp_video_file:
        temp_video_file.write(video_bytes)
        temp_video_path = temp_video_file.name

        # Run audio and video analysis in parallel for efficiency
        with ThreadPoolExecutor() as executor:
            speech_future = executor.submit(_process_audio_task, temp_video_path, speech_model)
            video_future = executor.submit(_process_video_task, temp_video_path, video_model)

            speech_result = speech_future.result() or "unknown"
            video_result = video_future.result() or "unknown"

    # Construct the detailed prompt for the RAG model
    # This combines all context for the most intelligent response
    combined_query = (
        f"Please act as a supportive assistant. Based on the user's input and emotional context, "
        f"provide a helpful and empathetic response.\n\n"
        f"--- Emotional Context ---\n"
        f"Detected Speech Tone: {speech_result}\n"
        f"Detected Facial Expression: {video_result}\n\n"
        f"--- User's Message ---\n"
        f"'{user_text if user_text else 'No text message provided.'}'"
    )

    # Get the final, context-aware response from the RAG assistant
    response = assistant.get_response(combined_query)
    return response.get('response_text', "Sorry, I couldn't generate a response at this time.")
