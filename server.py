import os
import logging
from fastapi import FastAPI, UploadFile, HTTPException
import jax.numpy as jnp
from transformers.pipelines.audio_utils import ffmpeg_read
from whisper_jax import FlaxWhisperPipline
from jax.experimental.compilation_cache import compilation_cache as cc
import numpy as np
import time

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize JAX cache and model configuration
cc.initialize_cache("./jax_cache")
checkpoint = "openai/whisper-large-v3"
BATCH_SIZE = 1
FILE_LIMIT_MB = 1000

# Initialize FastAPI app
app = FastAPI(title="Whisper JAX API")

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
    # Initialize pipeline
    pipeline = FlaxWhisperPipline(checkpoint, dtype=jnp.bfloat16, batch_size=BATCH_SIZE)
    
    # Pre-compile step to warm up the model
    logger.info("Compiling forward call...")
    start = time.time()
    random_inputs = {
        "input_features": np.ones(
            (BATCH_SIZE, pipeline.model.config.num_mel_bins, 2 * pipeline.model.config.max_source_positions)
        )
    }
    random_timestamps = pipeline.forward(random_inputs, batch_size=BATCH_SIZE, return_timestamps=True)
    compile_time = time.time() - start
    logger.info(f"Compiled in {compile_time}s")

    # Start server
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
