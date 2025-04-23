# PDF RAG System for macOS

A local Retrieval-Augmented Generation (RAG) system for querying your PDF library using Ollama and a local LLM. This system allows you to drop PDFs into a library folder and query them with natural language questions.

## Features

- Process PDFs only once, with tracking to avoid reprocessing
- Incremental processing of new PDFs without rebuilding the entire index
- Efficient storage of chunks and embeddings
- Query your PDF library with any question using Ollama for LLM inference
- No need to download large models directly - Ollama handles model management

## Project Structure

/
├── library/                   # Drop your PDFs here
├── processed/                 # Processed data is stored here
│   ├── chunks/                # Text chunks extracted from PDFs
│   ├── embeddings/            # Embeddings for each document
│   └── index/                 # FAISS index and metadata
├── processed_files.json       # Tracks which PDFs have been processed
├── project_structure.py       # Creates the folder structure
├── process_pdfs.py            # Processes PDFs in the library
├── query_rag_ollama.py        # Query the system with Ollama
└── requirements.txt           # Dependencies

## Setup Instructions

### 1. Install Ollama

Download and install Ollama from ollama.com/download

Navigate to the correct folder inside terminal

### 2. Pull a Model in Ollama

Open Terminal and pull a model:

# Start Ollama (if not already running)
ollama serve

# Pull a model (in another terminal window)
ollama pull llama3:8b

You can check available models with:
ollama list

### 3. Create a Virtual Environment

# Create a new directory for your project
mkdir pdf_rag_system
cd pdf_rag_system

# Create a virtual environment
python3 -m venv rag_env

# Activate the virtual environment
source rag_env/bin/activate

### 4. Install Dependencies

pip install sentence-transformers faiss-cpu PyPDF2 tiktoken requests numpy

### 5. Set Up Project Structure

python project_structure.py

### 6. Add PDFs to the Library

Place your PDF files in the "library" folder.

### 7. Process PDFs

python process_pdfs.py

This will:
- Extract text from PDFs that haven't been processed yet
- Chunk the text into manageable pieces
- Generate embeddings for the chunks
- Update the FAISS index

### 8. Query the System with Ollama

Make sure Ollama is running:
ollama serve

Then query your PDFs:
python query_rag_ollama.py "Your question about the PDFs" --model llama3:8b --temperature 0.2

## Advanced Usage

### Force Reprocessing of All PDFs

python process_pdfs.py --force

### Change Chunk Size

python process_pdfs.py --chunk-size 300

### Change Ollama Model

python query_rag_ollama.py "Your question" --model "mistral"

You must have already pulled the model with "ollama pull mistral" first.

### Using Other Ollama Models

Popular models to try:
- llama3:8b (recommended default)
- llama3
- mistral
- mixtral
- phi
- neural-chat

Pull any model with:
ollama pull model_name

### Change Number of Retrieved Chunks

python query_rag_ollama.py "Your question" --top-k 5

### Adjust Temperature

python query_rag_ollama.py "Your question" --temperature 0.2

## Troubleshooting

### Ollama Not Running
If you get a connection error, make sure Ollama is running with:
ollama serve

### Model Not Found
If you get a model not found error, pull the model first:
ollama pull model_name

### Memory Issues
If you experience memory issues:
- Try a smaller model (e.g., tinyllama, phi)
- Reduce the number of chunks retrieved (--top-k 2)
- Reduce the max tokens generated (--max-tokens 300)

## Extending the System

This system is designed to be easily extended. Some possible improvements:
- Add support for more document types (Word, HTML, etc.)
- Implement better chunking strategies (by paragraph, sliding window, etc.)
- Use a more sophisticated embedding model
- Add a simple web interface
- Implement hybrid search (combining keyword and semantic search)

## Requirements

- Python 3.8+
- macOS (should work on other platforms but tested on macOS)
- Ollama installed (from ollama.com)