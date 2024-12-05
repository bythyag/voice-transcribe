# AI-Powered Document Q&A Assistant

## Overview
An AI assistant that enables real-time, context-aware question answering by processing audio input and leveraging document context, web search, and AI-powered analysis.

## Features
- Real-time audio transcription using Whisper
- Document context extraction and relevance matching
- Web search integration for comprehensive answers
- Flexible context sourcing (document, web, or AI knowledge)
- Multi-threaded processing for efficient performance

## Prerequisites
- Python 3.8+
- API keys for:
  - OpenAI
  - Tavily

## Installation
```bash
git clone https://github.com/yourusername/ai-document-assistant.git
cd ai-document-assistant
pip install -r requirements.txt
```

## Environment Setup
Create a `.env` file with:
```
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## Usage
```bash
python main.py
```
1. Provide a PDF or TXT document path
2. Select an audio input device
3. Start asking questions about the document

## Key Dependencies
- OpenAI
- Tavily
- Faster Whisper
- PyAudio
- NumPy

## License
MIT License
