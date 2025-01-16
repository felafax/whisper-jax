import requests
import time
import statistics
from pathlib import Path

def measure_transcription_latency(file_path, num_loops=10):
    # API endpoint
    url = "http://localhost:8000/transcribe/"
    
    # Get audio duration
    import librosa
    audio_duration = librosa.get_duration(path=file_path)
    
    latencies = []
    rtfs = []
    
    print(f"Starting {num_loops} transcription tests...")
    print(f"Audio duration: {audio_duration:.2f} seconds")
    print("-" * 50)
    
    for i in range(num_loops):
        # Prepare the file for upload
        files = {
            'file': ('sample.mp3', open(file_path, 'rb'), 'audio/mpeg')
        }
        
        # Measure time
        start_time = time.time()
        
        # Make the request
        response = requests.post(url, files=files)
        
        end_time = time.time()
        
        if response.status_code == 200:
            elapsed_time = end_time - start_time
            rtf = audio_duration / elapsed_time
            
            latencies.append(elapsed_time)
            rtfs.append(rtf)
            
            print(f"Run {i+1:2d}: Latency = {elapsed_time:.2f}s, RTF = {rtf:.2f}x")
        else:
            print(f"Run {i+1:2d}: Failed with status code {response.status_code}")
            print(f"Error: {response.text}")
    
    # Calculate statistics
    if latencies:
        avg_latency = statistics.mean(latencies)
        avg_rtf = statistics.mean(rtfs)
        std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0
        std_rtf = statistics.stdev(rtfs) if len(rtfs) > 1 else 0
        
        print("\nResults:")
        print("-" * 50)
        print(f"Average Latency: {avg_latency:.2f}s ± {std_latency:.2f}s")
        print(f"Average RTF: {avg_rtf:.2f}x ± {std_rtf:.2f}x")
        print(f"Audio Duration: {audio_duration:.2f}s")
        
        return {
            'avg_latency': avg_latency,
            'avg_rtf': avg_rtf,
            'std_latency': std_latency,
            'std_rtf': std_rtf,
            'audio_duration': audio_duration
        }
    
    return None

if __name__ == "__main__":
    # Replace with your audio file path
    audio_file = "sample.mp3"
    
    if not Path(audio_file).exists():
        print(f"Error: Audio file '{audio_file}' not found!")
        exit(1)
    
    results = measure_transcription_latency(audio_file, num_loops=10)