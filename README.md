# 🤖 AI-Powered Document Q&A Assistant

## 🌟 Overview
An advanced AI assistant that transforms how you interact with documents. Leveraging cutting-edge AI technologies, this tool enables real-time, context-aware question answering by processing audio input and intelligently extracting insights from your documents.

## ✨ Features
- 🎙️ Real-time audio transcription with high accuracy
- 📄 Intelligent document context extraction
- 🌐 Seamless web search integration
- 🧠 Adaptive context sourcing (document, web, or AI knowledge)
- 🚀 Efficient multi-threaded processing

## 🛠️ Installation

### Prerequisites
- 🐍 Python 3.8+
- 🔑 API Keys:
  - OpenAI API Key
  - Tavily API Key

### Quick Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-document-assistant.git
cd ai-document-assistant
```

2. Install dependencies:
```bash
pip install openai tavily-python faster-whisper pyaudio numpy python-dotenv PyPDF2 sounddevice tiktoken
```

3. Create a `.env` file:
```
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## 🚀 Getting Started
1. Run the application
```bash
python main.py
```
2. When prompted, enter the full path to your PDF or TXT document
3. Select your audio input device
4. Start asking questions about the document!

## 🔧 Detailed Dependencies
- **🤖 OpenAI**: Powering intelligent text generation and embeddings
- **🌐 Tavily**: Providing real-time web search capabilities
- **🎙️ Faster Whisper**: Enabling rapid, accurate speech-to-text transcription
- **🔊 PyAudio**: Managing audio input streams
- **🧮 NumPy**: Supporting high-performance numerical computations

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License
MIT License

## 🛡️ Disclaimer
This tool is an AI-powered assistant and may occasionally provide imperfect responses. Always verify critical information.
