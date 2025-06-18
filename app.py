import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from pathlib import Path
import logging
from rag_query import RAGChatClient
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from typing import Optional, List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Chatbot API", description="API for multi-turn conversational RAG chatbot")

# Serve static files
app.mount("/static", StaticFiles(directory="."), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class UploadFileRequest(BaseModel):
    filename: str = Field(..., description="Name of the uploaded file")
    
    @validator('filename')
    def validate_filename(cls, v):
        if not v.lower().endswith('.pdf'):
            raise ValueError('File must be a PDF')
        return v

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query, cannot be empty")
    pdf_name: str = Field(..., min_length=1, description="PDF name without extension, cannot be empty")

class HistoryEntry(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$", description="Role of the message (user or assistant)")
    content: str = Field(..., description="Message content")

class ValidationErrorDetail(BaseModel):
    loc: list[str]
    msg: str
    type: str

class ValidationErrorResponse(BaseModel):
    detail: list[ValidationErrorDetail]

# Store RAG clients by PDF name
rag_clients = {}

# Custom exception handler for validation errors
@app.exception_handler(422)
async def validation_exception_handler(request: Request, exc: HTTPException):
    if isinstance(exc.detail, list):
        errors = [
            ValidationErrorDetail(
                loc=[str(loc) for loc in error["loc"]],
                msg=error["msg"],
                type=error["type"]
            )
            for error in exc.detail
        ]
        return JSONResponse(
            status_code=422,
            content=ValidationErrorResponse(detail=errors).dict()
        )
    return JSONResponse(
        status_code=422,
        content=ValidationErrorResponse(detail=[
            ValidationErrorDetail(loc=["unknown"], msg=str(exc.detail), type="value_error")
        ]).dict()
    )

@app.post("/upload-pdf", response_model=dict)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and index a PDF file"""
    try:
        UploadFileRequest(filename=file.filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    
    pdf_path = f"./uploads/{file.filename}"
    os.makedirs("./uploads", exist_ok=True)
    with open(pdf_path, "wb") as f:
        f.write(await file.read())
    
    pdf_name = Path(pdf_path).stem
    persist_dir = f"./chroma_db_{pdf_name}"
    
    try:
        from vector_utils import load_document, split_document, generate_embeddings, store_in_chromadb
        docs = load_document(pdf_path)
        chunks = split_document(docs)
        store_in_chromadb(chunks, generate_embeddings(), persist_dir)
        rag_clients[pdf_name] = RAGChatClient(persist_dir)
        logger.info(f"PDF {file.filename} indexed and RAG client initialized")
        return {"message": f"PDF {file.filename} indexed successfully"}
    except Exception as e:
        logger.error(f"Error indexing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error indexing PDF: {str(e)}")

@app.post("/query", response_model=dict)
async def post_query(request: QueryRequest):
    """Process user query for a specific PDF and return LLM response"""
    pdf_name = request.pdf_name
    query = request.query
    
    if pdf_name not in rag_clients:
        raise HTTPException(status_code=404, detail=f"No RAG client found for PDF: {pdf_name}")
    
    try:
        response = rag_clients[pdf_name].get_response(query)
        logger.info(f"Query processed for {pdf_name}: {query[:50]}... Response: {response[:50]}...")
        return {"query": query, "response": response}
    except Exception as e:
        logger.error(f"Error processing query for {pdf_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/history", response_model=list[HistoryEntry])
async def get_history(pdf_name: str = Query(..., min_length=1, description="PDF name without extension")):
    """Retrieve conversation history for a specific PDF"""
    if pdf_name not in rag_clients:
        raise HTTPException(status_code=404, detail=f"No RAG client found for PDF: {pdf_name}")
    
    history = [
        HistoryEntry(
            role="user" if isinstance(msg, HumanMessage) else "assistant",
            content=msg.content
        )
        for msg in rag_clients[pdf_name].conversation_history
    ]
    logger.info(f"Retrieved {len(history)} history entries for {pdf_name}")
    return history

@app.get("/list-pdfs", response_model=List[str])
async def list_pdfs():
    """List all indexed PDF names"""
    pdfs = list(rag_clients.keys())
    logger.info(f"Retrieved {len(pdfs)} indexed PDFs")
    return pdfs

@app.on_event("startup")
def startup_event():
    """Initialize default RAG client for existing vector stores"""
    load_dotenv()
    for persist_dir in Path("./").glob("chroma_db_*"):
        pdf_name = persist_dir.name.replace("chroma_db_", "")
        try:
            rag_clients[pdf_name] = RAGChatClient(str(persist_dir))
            logger.info(f"Initialized RAG client for {pdf_name}")
        except Exception as e:
            logger.error(f"Error initializing RAG client for {pdf_name}: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)