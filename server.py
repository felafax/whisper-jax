import os
import logging
from fastapi import FastAPI, UploadFile, HTTPException
import jax.numpy as jnp
from transformers.pipelines.audio_utils import ffmpeg_read
from whisper_jax import FlaxWhisperPipline
from jax.experimental.compilation_cache import compilation_cache as cc

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize JAX cache and model configuration
cc.initialize_cache("./jax_cache")
checkpoint = "openai/whisper-large-v3"
BATCH_SIZE = 32
FILE_LIMIT_MB = 1000

# Initialize FastAPI app
app = FastAPI(title="Whisper JAX API")

# Initialize the pipeline at startup
pipeline = FlaxWhisperPipline(checkpoint, dtype=jnp.bfloat16, batch_size=BATCH_SIZE)

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile):
    try:
        # Check file size
        contents = await file.read()
        file_size_mb = len(contents) / (1024 * 1024)
        
        if file_size_mb > FILE_LIMIT_MB:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds limit of {FILE_LIMIT_MB}MB. Got {file_size_mb:.2f}MB"
            )

        # Process audio
        inputs = ffmpeg_read(contents, pipeline.feature_extractor.sampling_rate)
        inputs = {"array": inputs, "sampling_rate": pipeline.feature_extractor.sampling_rate}

        # Generate transcription
        logger.info("Transcribing audio...")
        outputs = pipeline(inputs, batch_size=BATCH_SIZE, task="transcribe")
        text = outputs["text"]
        
        return {"transcription": text}

    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 