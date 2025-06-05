import os
import json
import queue
import threading
import numpy as np
import pyaudio
import sounddevice as sd

import PyPDF2
import tiktoken
import re

from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient
from faster_whisper import WhisperModel

# Load environment variables
load_dotenv()

# Initialize Clients and Models
openai_client = OpenAI()
tavily_client = TavilyClient(api_key=os.getenv('TAVILY_API_KEY'))
whisper_model = WhisperModel('small', device='cpu')

# Audio Configuration
CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000
SILENCE_THRESHOLD = 0.01
MIN_AUDIO_LENGTH = 0.5
WORD_LIMIT = 30
OVERLAP_WORDS = 10

# Embedding and RAG Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
MAX_SECTION_LENGTH = 1000
SIMILARITY_THRESHOLD = 0.75


class DocumentProcessor:
    def __init__(self, file_path):
        """
        Initialize DocumentProcessor with a file path.
        :param file_path: Full path to the PDF or TXT file
        :raises FileNotFoundError: If file does not exist.
        :raises ValueError: If file type is unsupported.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        self.file_path = file_path
        self.text = self.extract_text()
        self.chunks = self.split_text(max_section_length=MAX_SECTION_LENGTH)
        self.embeddings = self.create_embeddings()
        self.summary = self.get_document_summary()

    def extract_text(self):
        """Extract text from PDF or TXT file."""
        if self.file_path.lower().endswith('.pdf'):
            with open(self.file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                return " ".join(page.extract_text() for page in reader.pages if page.extract_text())
        elif self.file_path.lower().endswith('.txt'):
            with open(self.file_path, 'r', encoding='utf-8') as file:
                return file.read()
        else:
            raise ValueError("Unsupported file type. Please use PDF or TXT.")

    def split_text(self, max_section_length=1000):
        """Split text into overlapping chunks using tiktoken."""
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        tokens = encoding.encode(self.text)
        chunks = []
        for i in range(0, len(tokens), max_section_length):
            chunk = encoding.decode(tokens[i:i + max_section_length])
            chunks.append(chunk)
        return chunks

    def create_embeddings(self, embedding_model=EMBEDDING_MODEL):
        """Create embeddings for each text chunk."""
        embeddings = []
        for chunk in self.chunks:
            try:
                response = openai_client.embeddings.create(
                    input=chunk,
                    model=embedding_model
                )
                embeddings.append(response.data[0].embedding)
            except Exception as e:
                print(f"Embedding error for a chunk: {e}")
                embeddings.append(None)
        return embeddings

    def get_document_summary(self, max_length=5000):
        """Generate a summary of the document using OpenAI."""
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Provide a concise 3-4 sentence summary of the document."},
                    {"role": "user", "content": self.text[:max_length]}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating document summary: {e}")
            return ""

    def get_relevant_context(self, query, top_k=3, similarity_threshold=SIMILARITY_THRESHOLD):
        """Retrieve relevant text chunks based on similarity of embeddings."""
        try:
            query_embedding = openai_client.embeddings.create(
                input=query,
                model=EMBEDDING_MODEL
            ).data[0].embedding

            similarities = []
            for emb in self.embeddings:
                if emb is None:
                    similarities.append(0)
                else:
                    sim = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
                    similarities.append(sim)

            top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_k]
            relevant_contexts = [self.chunks[i] for i in top_indices if similarities[i] >= similarity_threshold]
            return relevant_contexts
        except Exception as e:
            print(f"Error in retrieving relevant context: {e}")
            return []


def check_query_relevance(query, document_summary, threshold=0.5):
    """
    Check if the query is relevant to the document summary using OpenAI.
    :param query: The extracted query text.
    :param document_summary: Summary of the document.
    :param threshold: Relevance threshold.
    :return: True if relevant, False otherwise.
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    "Assess the relevance of the query to the document summary. "
                    "Return a float between 0 and 1 representing the relevance."
                )},
                {"role": "user", "content": f"Document Summary: {document_summary}\n\nQuery: {query}"}
            ]
        )
        relevance_score = float(response.choices[0].message.content.strip())
        print(f"\nQuery Relevance Score: {relevance_score}")
        return relevance_score >= threshold
    except Exception as e:
        print(f"Error checking query relevance: {e}")
        return False


def extract_query(text):
    """
    Extract potential query from the transcribed text using OpenAI.
    :param text: Transcribed text.
    :return: Extracted query string.
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    "Extract the main topic or question from the given text. "
                    "Return ONLY the extracted topic."
                )},
                {"role": "user", "content": text}
            ]
        )
        query = response.choices[0].message.content.strip()
        print(f"\nExtracted Query: {query}")
        return query if query and len(query) > 3 else None
    except Exception as e:
        print(f"Error extracting query: {e}")
        return None


def tavily_search(query, max_results=5):
    """
    Perform web search using the Tavily API.
    :param query: Search query.
    :param max_results: Maximum number of search results.
    :return: Formatted string containing search results.
    """
    try:
        search_results = tavily_client.search(query=query, max_results=max_results, include_answer=True)
        context = "\n\nWeb Search Context:\n"
        for result in search_results.get('results', []):
            context += f"Title: {result.get('title', 'N/A')}\n"
            context += f"URL: {result.get('url', 'N/A')}\n"
            context += f"Content: {result.get('content', 'N/A')}\n\n"
        return context
    except Exception as e:
        print(f"Tavily Search Error: {e}")
        return ""


def process_with_openai(text, document_processor=None):
    """
    Process text with OpenAI. Incorporates document context or web search for an answer.
    :param text: Input text.
    :param document_processor: DocumentProcessor instance (optional).
    :return: Dictionary containing questions, answers, and source.
    """
    try:
        query = extract_query(text)
        context = ""
        source = "knowledge"

        if document_processor and query:
            if check_query_relevance(query, document_processor.summary):
                relevant_contexts = document_processor.get_relevant_context(query)
                if relevant_contexts:
                    context = "\n\nRelevant Document Context:\n" + "\n".join(relevant_contexts)
                    source = "document"
            else:
                context = tavily_search(query)
                if context:
                    source = "web"

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    "You are an advanced AI assistant. Generate a JSON response with a 'questions' array. "
                    "Each element should be an object with 'question', 'answer', and 'source'. "
                    "If no clear question is found, return an empty 'questions' array. "
                    "Answer in 30 words or less."
                )},
                {"role": "user", "content": text + context}
            ],
            response_format={"type": "json_object"}
        )
        raw_response = response.choices[0].message.content.strip()
        try:
            result = json.loads(raw_response)
            # Update source for all provided questions
            for qa in result.get('questions', []):
                qa['source'] = source
            return result
        except json.JSONDecodeError as json_err:
            print(f"JSON Parsing Error: {json_err}")
            return {"questions": []}
    except Exception as e:
        print(f"Error processing with OpenAI: {e}")
        return {"questions": []}


class AudioProcessor:
    def __init__(self, device_index, document_processor=None):
        self.audio_queue = queue.Queue()
        self.transcript_queue = queue.Queue()
        self.device_index = device_index
        self.document_processor = document_processor
        self.audio_thread = None
        self.processing_thread = None

    def start(self):
        """Start audio transcription and query processing workers."""
        self.audio_thread = threading.Thread(
            target=self.transcription_worker, args=(), daemon=True
        )
        self.processing_thread = threading.Thread(
            target=self.question_processing_worker, args=(), daemon=True
        )
        self.audio_thread.start()
        self.processing_thread.start()
        self.capture_audio()

    def transcription_worker(self):
        """
        Worker thread for transcribing audio chunks.
        Uses faster_whisper model to transcribe audio data.
        """
        while True:
            audio_data = self.audio_queue.get()
            if audio_data is None:
                break
            try:
                segments, _ = whisper_model.transcribe(audio_data, language='en')
                transcript = " ".join(segment.text for segment in segments).strip()
                if transcript:
                    print(transcript, end=' ', flush=True)
                    self.transcript_queue.put(transcript)
            except Exception as e:
                print(f"Transcription error: {e}")

    def question_processing_worker(self):
        """
        Worker thread for processing transcripts.
        Accumulates transcript until a word limit is reached, then processes the text.
        """
        current_transcript = ""
        while True:
            transcript = self.transcript_queue.get()
            if transcript is None:
                break
            current_transcript = (current_transcript + " " + transcript).strip()
            if len(current_transcript.split()) >= WORD_LIMIT:
                query = extract_query(current_transcript)
                if query:
                    result = process_with_openai(query, self.document_processor)
                    if result.get("questions"):
                        for qa in result["questions"]:
                            if qa.get('question') and qa.get('answer'):
                                print(f"\nQ: {qa['question']}")
                                print(f"A: {qa['answer']}")
                                print(f"Source: {qa['source']}\n")
                    else:
                        print("\nNo questions detected.\n")
                # Retain last few words for overlapping context
                current_transcript = " ".join(current_transcript.split()[-OVERLAP_WORDS:])

    @staticmethod
    def list_audio_devices():
        """List available audio input devices."""
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            print(f"Device {i}: {device['name']}")

    def capture_audio(self):
        """Capture audio from the selected input device using PyAudio."""
        p = pyaudio.PyAudio()
        try:
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=CHUNK
            )
            print("\nStarted listening...\n")
            audio_buffer = []
            while True:
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                except Exception as e:
                    print(f"Audio stream read error: {e}")
                    continue
                audio = np.frombuffer(data, np.float32)
                if np.max(np.abs(audio)) > SILENCE_THRESHOLD:
                    audio_buffer.append(audio)
                    if len(audio_buffer) * CHUNK / RATE >= MIN_AUDIO_LENGTH:
                        combined_audio = np.concatenate(audio_buffer)
                        self.audio_queue.put(combined_audio)
                        audio_buffer = []
        except KeyboardInterrupt:
            print("\n\nStopping audio capture...")
        except Exception as e:
            print(f"Error in capturing audio: {e}")
        finally:
            self.audio_queue.put(None)
            self.transcript_queue.put(None)
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
            p.terminate()


def main():
    """Main function to start document processing and audio transcription."""
    document_path = input("Enter the full path to your PDF or TXT document: ").strip()
    document_processor = None
    try:
        document_processor = DocumentProcessor(document_path)
        print("\nDocument Summary:")
        print(document_processor.summary)
    except Exception as e:
        print(f"Error processing document: {e}")

    AudioProcessor.list_audio_devices()
    try:
        device_index = int(input("Select input device index: "))
    except ValueError:
        print("Invalid device index. Exiting.")
        return

    audio_processor = AudioProcessor(device_index, document_processor)
    audio_processor.start()


if __name__ == "__main__":
    main()
