import pyaudio
import numpy as np
from faster_whisper import WhisperModel

# Initialize Faster Whisper
model = WhisperModel('base', device='cpu')

# Audio configuration
CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000
SILENCE_THRESHOLD = 0.01
MIN_AUDIO_LENGTH = 2  # seconds
AUDIO_BUFFER = []

# Initialize PyAudio
p = pyaudio.PyAudio()

def list_audio_devices():
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
        device_info = p.get_device_info_by_index(i)
        print(f"Device {i}: {device_info['name']}")

try:
    # Select input device
    list_audio_devices()
    device_index = int(input("Select input device index: "))
    
    # Open audio stream
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=CHUNK
    )
    
    print("Started listening...\n")
    
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio = np.frombuffer(data, np.float32)
        
        if np.max(np.abs(audio)) > SILENCE_THRESHOLD:
            AUDIO_BUFFER.append(audio)
            
            if len(AUDIO_BUFFER) * CHUNK / RATE >= MIN_AUDIO_LENGTH:
                combined_audio = np.concatenate(AUDIO_BUFFER)
                
                try:
                    segments, _ = model.transcribe(combined_audio, language='en')
                    text = "".join([segment.text for segment in segments])
                    
                    if text.strip():
                        print(text.strip(), end=' ', flush=True)
                except Exception as e:
                    print(f"\nTranscription error: {str(e)}")
                
                AUDIO_BUFFER = []

except KeyboardInterrupt:
    print("\n\nStopping...")
except Exception as e:
    print(f"\nError: {str(e)}")
finally:
    if 'stream' in locals():
        stream.stop_stream()
        stream.close()
    p.terminate()