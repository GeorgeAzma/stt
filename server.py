import os
import time
import asyncio
import tempfile
import torch
import nemo.collections.asr as nemo_asr
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, HTTPException, Header, File, Form, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

model = None
last_access_time: float = 0.0
MODEL_UNLOAD_TIMEOUT_SECONDS = 5 * 60  # 5 minutes of inactivity


async def get_model():
    global model, last_access_time
    if model is None:
        print("Loading ASR model...")
        model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/parakeet-tdt-0.6b-v3"
        )
        print("ASR model loaded.")
    last_access_time = time.time()
    return model


async def unload_model_periodically():
    """Periodically checks for model inactivity and unloads it."""
    global model, last_access_time
    while True:
        await asyncio.sleep(60)
        if model is not None:
            current_time = time.time()
            if current_time - last_access_time > MODEL_UNLOAD_TIMEOUT_SECONDS:
                print(
                    f"Model inactive for {MODEL_UNLOAD_TIMEOUT_SECONDS/60} minutes. Unloading model..."
                )
                if torch.cuda.is_available():
                    model.to("cpu")
                    torch.cuda.empty_cache()
                del model
                model = None
                print("NeMo ASR model unloaded and CUDA memory cleared.")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manages application startup and shutdown events."""
    print("Application starting up...")
    asyncio.create_task(unload_model_periodically())
    yield
    print("Application shutting down...")


app = FastAPI(lifespan=lifespan)

@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(...),
    model: str = Form("parakeet-tdt-0.6b-v3"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    authorization: Optional[str] = Header(None),
):
    """OpenAI API compatible endpoint for speech-to-text transcription."""
    global last_access_time

    # Validate file presence and type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No selected file")

    # Check file size (50MB limit)
    MAX_SIZE = 50 * 1024 * 1024  # 50MB
    file_size = 0
    chunk_size = 1024 * 1024  # 1MB chunks

    contents = b""
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        file_size += len(chunk)
        if file_size > MAX_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds maximum limit of {MAX_SIZE/(1024*1024)}MB",
            )
        contents += chunk

    # Reset file pointer for later use
    await file.seek(0)

    temp_audio_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=f".{file.filename.split('.')[-1]}",
        ) as temp_audio_file:
            temp_audio_file.write(contents)
            temp_audio_path = temp_audio_file.name

        try:
            model = await asyncio.wait_for(get_model(), timeout=30.0)
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=503,
                detail="Model loading timed out. Please try again later.",
            )
        except Exception as model_error:
            print(f"Model initialization error: {str(model_error)}")
            raise HTTPException(
                status_code=500,
                detail="Failed to initialize speech recognition model. Please try again.",
            )

        if model is None:
            raise HTTPException(
                status_code=500,
                detail="Speech recognition model failed to initialize. Please try again.",
            )

        last_access_time = time.time()
        transcriptions = model.transcribe([temp_audio_path])

        if transcriptions:
            transcription_text = transcriptions[0].text
            return JSONResponse({"text": transcription_text})
        else:
            return JSONResponse(
                status_code=422,
                content={
                    "text": "",
                    "warning": "Could not transcribe audio. File may be empty or corrupted.",
                },
            )

    except Exception as e:
        error_detail = str(e)
        error_type = type(e).__name__
        print(f"Error during transcription: {error_type} - {error_detail}")

        if "'NoneType' object has no attribute" in error_detail:
            error_message = (
                "Failed to initialize speech recognition model. Please try again."
            )
        elif "CUDA" in error_detail:
            error_message = "GPU processing error. Please try again."
        elif "memory" in error_detail.lower():
            error_message = (
                "Server memory limit exceeded. Please try with a smaller file."
            )
        else:
            error_message = (
                "Failed to process audio file. Please ensure it's a valid audio file."
            )

        raise HTTPException(status_code=500, detail=error_message)

    finally:
        # Clean up the temporary file
        if temp_audio_path:
            try:
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
            except Exception as e:
                print(f"Failed to cleanup temporary file {temp_audio_path}: {e}")


@app.post("/v1/model/unload")
async def unload_model():
    global model
    if model is not None:
        if torch.cuda.is_available():
            model.to("cpu")
            torch.cuda.empty_cache()
        del model
        model = None
        return {"message": "Model unloaded and CUDA memory cleared successfully."}
    else:
        return {"message": "Model is already unloaded."}

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5001)
