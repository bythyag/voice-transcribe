import os
import json
import numpy as np
import pyaudio
import queue
from threading import Thread
from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient
from faster_whisper import WhisperModel
import PyPDF2
import tiktoken
import re
import sounddevice as sd
import numpy as np

# Load environment variables
load_dotenv()

# Initialize OpenAI Client
openai_client = OpenAI()

# Initialize Tavily Client
tavily_client = TavilyClient(api_key=os.getenv('TAVILY_API_KEY'))

# Initialize Faster Whisper
model = WhisperModel('small', device='cpu')

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
        :raises ValueError: If file does not exist or is unsupported
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        
        self.file_path = file_path
        self.text = self.extract_text()
        self.chunks = self.split_text()
        
        self.client = openai_client
        self.embeddings = self.create_embeddings()
        self.summary = self.get_document_summary()

    def extract_text(self):
        """Extract text from PDF or TXT file."""
        if self.file_path.lower().endswith('.pdf'):
            with open(self.file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                return " ".join([page.extract_text() for page in pdf_reader.pages])
        elif self.file_path.lower().endswith('.txt'):
            with open(self.file_path, 'r', encoding='utf-8') as file:
                return file.read()
        else:
            raise ValueError("Unsupported file type. Please use PDF or TXT.")

    def split_text(self, max_section_length=1000):
        """Split text into overlapping chunks."""
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        chunks = []
        tokens = encoding.encode(self.text)
        
        for i in range(0, len(tokens), max_section_length):
            chunk = encoding.decode(tokens[i:i+max_section_length])
            chunks.append(chunk)
        
        return chunks

    def create_embeddings(self, embedding_model="text-embedding-3-small"):
        """Create embeddings for text chunks."""
        embeddings = []
        for chunk in self.chunks:
            response = self.client.embeddings.create(
                input=chunk,
                model=embedding_model
            )
            embeddings.append(response.data[0].embedding)
        return embeddings

    def get_document_summary(self, max_length=5000):
        """Generate a summary of the document."""
        summary_response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Provide a concise 3-4 sentence summary of the document."},
                {"role": "user", "content": self.text[:max_length]}
            ]
        )
        return summary_response.choices[0].message.content

    def get_relevant_context(self, query, top_k=3, similarity_threshold=0.75):
        """Retrieve relevant context for a given query."""
        query_embedding = self.client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        ).data[0].embedding

        similarities = [
            np.dot(query_embedding, chunk_embedding) / 
            (np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding))
            for chunk_embedding in self.embeddings
        ]

        top_indices = sorted(
            range(len(similarities)), 
            key=lambda i: similarities[i], 
            reverse=True
        )[:top_k]

        relevant_contexts = [self.chunks[i] for i in top_indices if similarities[i] >= similarity_threshold]
        
        return relevant_contexts

def check_query_relevance(query, document_summary, threshold=0.5):
    """
    Check if the query is relevant to the document summary.
    
    :param query: The extracted query
    :param document_summary: Summary of the document
    :param threshold: Similarity threshold to consider the query relevant
    :return: Boolean indicating query relevance
    """
    try:
        relevance_response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    "Assess the relevance of the query to the document summary. "
                    "Return a float between 0 and 1 representing the relevance, "
                    "where 1 is highly relevant and 0 is not relevant at all."
                )},
                {"role": "user", "content": f"Document Summary: {document_summary}\n\nQuery: {query}"}
            ]
        )
        
        relevance_score = float(relevance_response.choices[0].message.content.strip())
        
        print(f"\nQuery Relevance Score: {relevance_score}")
        
        return relevance_score >= threshold
    
    except Exception as e:
        print(f"Error checking query relevance: {e}")
        return False

def extract_query(text):
    """
    Extract potential queries from transcribed text.
    
    :param text: Transcribed text
    :return: Extracted query or None
    """
    try:
        query_response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    "Extract the main topic or question from the given text. "
                    "If the text is a greeting or small talk, return a small talk topic. "
                    "If no clear topic is present, return a generic prompt. "
                    "Return ONLY the extracted topic, nothing else."
                )},
                {"role": "user", "content": text}
            ]
        )
        query = query_response.choices[0].message.content.strip()
        
        print("\nExtracted Query:", query)
        
        return query if query and len(query) > 3 else None
    except Exception as e:
        print(f"Error extracting query: {e}")
        return None

def tavily_search(query, max_results=5):
    """
    Perform web search using Tavily API.
    
    :param query: Search query
    :param max_results: Maximum number of search results
    :return: Formatted search results
    """
    try:
        search_results = tavily_client.search(
            query=query, 
            max_results=max_results,
            include_answer=True
        )
        
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
    Process text with OpenAI, using document context or web search.
    
    :param text: Input text to process
    :param document_processor: DocumentProcessor instance
    :return: Processed questions and answers
    """
    try:
        query = extract_query(text)
        
        context = ""
        source = "knowledge"
        
        if document_processor and query:
            is_relevant = check_query_relevance(query, document_processor.summary)
            
            if is_relevant:
                relevant_contexts = document_processor.get_relevant_context(query)
                context = "\n\nRelevant Document Context:\n" + "\n".join(relevant_contexts)
                source = "document"
            else:
                # If not relevant to document, try web search
                context = tavily_search(query)
                source = "web" if context else "knowledge"

        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    "You are an advanced AI assistant. Carefully analyze the input text. "
                    "If the text contains a clear question or topic for discussion, "
                    "generate a response. Use the provided context to inform your answer. "
                    "Follow these guidelines:"
                    "1. Return a JSON with a 'questions' array, where each object has:"
                    "   - 'question': The detected question or topic"
                    "   - 'answer': A comprehensive response"
                    "   - 'source': Either 'knowledge', 'document', or 'web'"
                    "2. If no clear question is found, return an empty 'questions' array."
                    "3. Answer in 30 words or less."
                )},
                {"role": "user", "content": text + context}
            ],
            response_format={"type": "json_object"}
        )
        
        raw_response = completion.choices[0].message.content
        
        try:
            result = json.loads(raw_response)
            
            # Update source for all questions
            for qa in result.get('questions', []):
                qa['source'] = source
            
            return result
        except json.JSONDecodeError as json_err:
            print(f"\nJSON Parsing Error: {json_err}")
            return {"questions": []}
    
    except Exception as e:
        print("\nError processing with OpenAI:", e)
        import traceback
        traceback.print_exc()
        return {"questions": []}

def transcription_worker(audio_queue, transcript_queue):
    """
    Worker thread for transcribing audio chunks.
    
    :param audio_queue: Queue for audio chunks
    :param transcript_queue: Queue for transcribed text
    """
    while True:
        audio_data = audio_queue.get()
        if audio_data is None:
            break
        
        segments, _ = model.transcribe(audio_data, language='en')
        transcript = " ".join(segment.text for segment in segments).strip()
        
        if transcript:
            print(transcript, end=' ', flush=True)
            transcript_queue.put(transcript)

def question_processing_worker(transcript_queue, document_processor):
    """
    Worker thread for processing transcripts and generating questions.
    
    :param transcript_queue: Queue for transcribed text
    :param document_processor: DocumentProcessor instance
    """
    current_transcript = ""
    while True:
        transcript = transcript_queue.get()
        if transcript is None:
            break
        
        current_transcript = (current_transcript + " " + transcript).strip()
        
        if len(current_transcript.split()) >= WORD_LIMIT:
            query = extract_query(current_transcript)
            
            if query:
                result = process_with_openai(query, document_processor)
                
                if result.get("questions"):
                    for qa in result["questions"]:
                        if qa.get('question') and qa.get('answer'):
                            print(f"\nQ: {qa['question']}")
                            print(f"A: {qa['answer']}")
                            print(f"Source: {qa['source']}\n")
                else:
                    print("\nNo questions detected.\n")
            
            current_transcript = " ".join(current_transcript.split()[-OVERLAP_WORDS:])

def list_audio_devices():
    """List available audio input devices."""
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        print(f"Device {i}: {device['name']}")

def main(document_path):
    """
    Main function to set up document processing and audio transcription.
    
    :param document_path: Path to input document
    """
    try:
        document_processor = DocumentProcessor(document_path)
        print("\nDocument Summary:")
        print(document_processor.summary)
    except Exception as e:
        print(f"Error processing document: {e}")
        return
    
    list_audio_devices()
    device_index = int(input("Select input device index: "))
    
    audio_queue = queue.Queue()
    transcript_queue = queue.Queue()
    
    transcription_thread = Thread(target=transcription_worker, args=(audio_queue, transcript_queue), daemon=True)
    question_thread = Thread(target=question_processing_worker, args=(transcript_queue, document_processor), daemon=True)
    transcription_thread.start()
    question_thread.start()
    
    p = pyaudio.PyAudio()
    
    try:
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK
        )
        
        print("\nStarted listening...\n")
        
        audio_buffer = []
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio = np.frombuffer(data, np.float32)
            
            if np.max(np.abs(audio)) > SILENCE_THRESHOLD:
                audio_buffer.append(audio)
                
                if len(audio_buffer) * CHUNK / RATE >= MIN_AUDIO_LENGTH:
                    combined_audio = np.concatenate(audio_buffer)
                    audio_queue.put(combined_audio)
                    audio_buffer = []
    
    except KeyboardInterrupt:
        print("\n\nStopping...")
    except Exception as e:
        print(f"\nError: {str(e)}")
    finally:
        audio_queue.put(None)
        transcript_queue.put(None)
        
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        p.terminate()

if __name__ == "__main__":
    
    document_path = input("Enter the full path to your PDF or TXT document: ")
    main(document_path)
