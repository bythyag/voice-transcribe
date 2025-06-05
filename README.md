## ai document q&a assistant

an ai assistant to live transcribe the converversation and use internet search, and document search (rag) to provide context to the transcript and answer questions if any.

### features
- real-time speech-to-text with faster whisper.
- voice query extraction.
- context retrieval from pdf/txt files.
- web search fallback when needed.
- multi-threaded audio and query processing.

### prerequisites
- python 3.8+
- api keys for openai and tavily

### setup
1. clone the repository and change to its directory:
```bash
git clone https://github.com/yourusername/ai-document-assistant.git
cd ai-document-assistant
````

2. install dependencies:

```bash
pip install openai tavily-python faster-whisper pyaudio numpy python-dotenv pypdf2 sounddevice tiktoken
```

3. create a `.env` file with your credentials:

```
openai_api_key=your_openai_api_key
tavily_api_key=your_tavily_api_key
```

### running the assistant

1. launch the app:
```bash
python main.py
```
2. provide the full path to your pdf or txt document.
3. pick an audio input device.
4. speak your query and receive concise answers
