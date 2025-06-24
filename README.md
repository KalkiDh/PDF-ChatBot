RAG Chatbot API
A Retrieval-Augmented Generation (RAG) chatbot API built with FastAPI, LangChain, ChromaDB, and Azure AI for processing and querying PDF documents.
Overview
This project implements a conversational chatbot that:

Processes uploaded PDF documents
Generates embeddings using Azure AI's text-embedding model
Stores document chunks in ChromaDB for vector-based retrieval
Provides a RESTful API for multi-turn conversations
Supports querying specific PDFs and retrieving conversation history

Features

PDF Processing: Upload and index PDF files for content retrieval
Vector Search: Uses ChromaDB for efficient similarity search
Conversational AI: Powered by Azure AI's chat completions model (Grok-3)
API Endpoints:
Upload PDFs (/upload-pdf)
Query documents (/query)
Retrieve conversation history (/history)
List indexed PDFs (/list-pdfs)


Error Handling: Robust logging and validation
CORS Support: Configured for cross-origin requests

Tech Stack

Backend: FastAPI (Python)
Document Processing: LangChain, UnstructuredLoader
Vector Store: ChromaDB
Embeddings & LLM: Azure AI (text-embedding-3-small, Grok-3)
Environment Management: python-dotenv
Logging: Python logging
Validation: Pydantic

Prerequisites

Python 3.8+
Azure AI account with API credentials
GitHub token (used as Azure authentication)
Required Python packages:pip install fastapi uvicorn langchain langchain-unstructured langchain-chroma azure-ai-inference python-dotenv pydantic



Setup

Clone the repository:
git clone <repository-url>
cd <repository-directory>


Install dependencies:
pip install -r requirements.txt


Set up environment variables:Create a .env file in the project root directory:
GITHUB_TOKEN=your_azure_ai_token


Run the API:
uvicorn main:app --host 0.0.0.0 --port 8000



Project Structure
├── vector_utils.py    # Document loading, embedding, and ChromaDB storage
├── rag_query.py       # RAG client logic for querying and conversation history
├── api.py             # FastAPI application with API endpoints
├── uploads/           # Directory for uploaded PDFs (auto-created)
├── chroma_db_*/       # ChromaDB vector stores (auto-created per PDF)
├── .env               # Environment variables
└── README.md          # Project documentation

API Usage
1. Upload a PDF
curl -X POST -F "file=@/path/to/document.pdf" http://localhost:8000/upload-pdf

2. Query a PDF
curl -X POST -H "Content-Type: application/json" -d '{"query":"Summarize this file","pdf_name":"document"}' http://localhost:8000/query

3. Get Conversation History
curl -X GET "http://localhost:8000/history?pdf_name=document"

4. List Indexed PDFs
curl -X GET http://localhost:8000/list-pdfs

Notes

PDFs are stored in ./uploads and vector stores in ./chroma_db_* directories.
All temporary files are deleted on server shutdown.
Ensure your Azure AI token is valid and has access to the specified models.
The API is configured for development with open CORS; restrict origins in production.

Contributing

Fork the repository
Create a feature branch (git checkout -b feature-name)
Commit changes (git commit -m "Add feature")
Push to the branch (git push origin feature-name)
Open a Pull Request

License
MIT License
