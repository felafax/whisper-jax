# API Server Documentation

## Core Components

- `server.py` - Main API server
- `whisper-docker-helper.sh` - Docker container management script for Whisper JAX server
  - Usage: `./whisper-docker-helper.sh push` to build and push latest image to GCR

## Deployment

### TPU Server Setup
```bash
sudo docker run --rm -it \
  --privileged \
  --network host \
  gcr.io/felafax-training/whisper-jax-server:latest
```

### TPU Firewall Configuration
```bash
gcloud compute tpus tpu-vm update "calm-bear-569" \
  --zone="europe-west4-b" \
  --project="felafax-training" \
  --add-tags=vllm-server
```

## Testing

### Test Files
```bash
wget https://storage.googleapis.com/felafax-public/sample.mp3
wget https://storage.googleapis.com/felafax-public/clip_10mins.mp3
```

### Performance Testing
```bash
pip install -r requirements-dev.txt
python perf.py sample.mp3
python perf.py clip_10mins.mp3
```

### API Testing
```bash
curl -X POST "http://<tpu-ip>:8000/transcribe/" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample.mp3"
```
