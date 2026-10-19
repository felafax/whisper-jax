# Whisper JAX

Open-source speech transcription with OpenAI Whisper and JAX. Runs on CPU, GPU, and TPU.

![Audio is decoded and chunked, transcribed by Whisper JAX, then returned as text.](docs/architecture.svg)

## Quick start

Requires JAX for your hardware, PyTorch, and FFmpeg.

```bash
pip install git+https://github.com/felafax/whisper-jax.git
```

```python
from whisper_jax import FlaxWhisperPipline

transcribe = FlaxWhisperPipline("openai/whisper-large-v3-turbo")
print(transcribe("audio.mp3")["text"])
```

[HTTP API & deployment](README-SERVER.md)
