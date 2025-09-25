import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import sys
from fastapi.middleware.cors import CORSMiddleware

# --- Path Setup ---
# Add the project's root directory to the Python path
# This allows us to import from the 'pipeline' module
sys.path.insert(0, str(Path(__file__).resolve().parent))

# --- Corrected Imports from your project ---
## CHANGE: We now import the single, powerful function from our pipeline.
from pipeline import process_multimodal_input, assistant
# Placeholder for your STT model - this part is fine.
# from models.stt import transcribe_audio_from_file

app = FastAPI()

# --- Middleware (CORS) ---
# This is set up correctly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for simplicity, can be restricted later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration ---
# Assuming your directory structure is:
# - client/
#   - index.html
#   - static/
#     - styles.css
#     - script.js
# - server/
#   - main.py
#   - pipeline.py
#   - models/
#   - temp_uploads/
frontend_dir = Path(__file__).resolve().parent.parent / "client"
temp_dir = Path(__file__).resolve().parent / "temp_uploads"
os.makedirs(temp_dir, exist_ok=True)

# --- Static Files ---
# Serve static assets like CSS and JS from a 'static' subdirectory
app.mount("/static", StaticFiles(directory=frontend_dir / "static"), name="static")


# --- Pydantic Models (Correctly Defined) ---
class AnalysisResponse(BaseModel):
    final_response: str

class ChatMessage(BaseModel):
    message: str

# --- API Endpoints ---

@app.post("/api/process_multimodal", response_model=AnalysisResponse)
async def process_multimodal_session(
    video: UploadFile = File(...),
    ## CHANGE: Added user_text from the form. It can be an empty string.
    user_text: str = Form("")
):
    """
    This single endpoint now handles the entire multimodal analysis.
    """
    try:
        # Read the video file from the request into memory
        video_bytes = await video.read()

        ## CHANGE: The entire logic is now just one clean function call.
        # This function handles temp files, audio extraction, parallel processing, and cleanup.
        final_response_text = process_multimodal_input(
            video_bytes=video_bytes,
            user_text=user_text
        )

        if "Error" in final_response_text:
            raise HTTPException(status_code=500, detail=final_response_text)

        return AnalysisResponse(final_response=final_response_text)

    except Exception as e:
        print(f"Error in process_multimodal endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@app.post("/api/chatbot_text", summary="Get a response from the text-only RAG chatbot")
async def chat_with_bot(request: ChatMessage):
    """
    Handles text-only conversations with the RAG assistant.
    """
    try:
        # This endpoint correctly uses the assistant directly.
        response = assistant.get_response(request.message)
        response_text = response.get('response_text', "Sorry, I couldn't process that.")
        return {"response": response_text}
    except Exception as e:
        print(f"Error getting chatbot response: {e}")
        raise HTTPException(status_code=500, detail="Chatbot response failed.")

# We no longer need the separate /transcribe_audio endpoint if it's part of the main flow.
# If you still need it for a different feature, you can keep it.

# --- Frontend Entrypoint ---
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_frontend():
    """
    Serves the main index.html file for the frontend application.
    """
    return FileResponse(frontend_dir / "index.html")

# --- Server Startup ---
if __name__ == "__main__":
    import uvicorn
    # Make sure to run from the 'server' directory
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")
