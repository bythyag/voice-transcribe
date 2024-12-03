import flask
from flask import Flask, render_template, request, jsonify, Response
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import pyaudio
import numpy as np
import threading
import queue
import wave
import io

app = Flask(__name__)

class RealTimeTranscriber:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.audio_queue = queue.Queue()
        self.transcription_queue = queue.Queue()
        self.is_recording = False
        self.chunk_size = 16000  # 1 second of audio at 16kHz
        self.sample_rate = 16000

    def audio_capture_thread(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=self.sample_rate,
                        input=True,
                        frames_per_buffer=self.chunk_size)
        
        while self.is_recording:
            data = stream.read(self.chunk_size)
            audio_chunk = np.frombuffer(data, dtype=np.int16)
            self.audio_queue.put(audio_chunk)

        stream.stop_stream()
        stream.close()
        p.terminate()

    def transcription_thread(self):
        accumulated_audio = []
        while self.is_recording or not self.audio_queue.empty():
            try:
                chunk = self.audio_queue.get(timeout=1)
                accumulated_audio.extend(chunk)

                # Process every 3 seconds
                if len(accumulated_audio) >= self.sample_rate * 3:
                    # Convert to numpy array
                    audio_data = np.array(accumulated_audio[:self.sample_rate * 3])
                    
                    # Save to temporary file or process directly
                    with io.BytesIO() as wav_io:
                        with wave.open(wav_io, 'wb') as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(self.sample_rate)
                            wf.writeframes(audio_data.tobytes())
                        
                        wav_io.seek(0)
                        result = self.model(wav_io.getvalue())
                        self.transcription_queue.put(result['text'])

                    # Keep only the last second to maintain overlap
                    accumulated_audio = accumulated_audio[self.sample_rate * 2:]

            except queue.Empty:
                continue

    def start_recording(self):
        self.is_recording = True
        self.audio_thread = threading.Thread(target=self.audio_capture_thread)
        self.transcribe_thread = threading.Thread(target=self.transcription_thread)
        
        self.audio_thread.start()
        self.transcribe_thread.start()

    def stop_recording(self):
        self.is_recording = False
        self.audio_thread.join()
        self.transcribe_thread.join()

# Global model setup (similar to your existing code)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "distil-whisper/distil-small.en"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

transcriber = RealTimeTranscriber(
    pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        torch_dtype=torch_dtype,
        device=device,
    ),
    processor
)
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/index')
def index():
    return render_template('index.html')
@app.route('/history')
def history():
    return render_template('history.html')
@app.route('/settings')
def settings():
    return render_template('settings.html')

@app.route('/start_recording', methods=['POST'])
def start_recording():
    transcriber.start_recording()
    return jsonify({"status": "Recording started"})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    transcriber.stop_recording()
    return jsonify({"status": "Recording stopped"})

@app.route('/transcription_stream')
def transcription_stream():
    def generate():
        while transcriber.is_recording or not transcriber.transcription_queue.empty():
            try:
                transcription = transcriber.transcription_queue.get(timeout=1)
                yield f"data: {transcription}\n\n"
            except queue.Empty:
                yield "data: \n\n"

    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)