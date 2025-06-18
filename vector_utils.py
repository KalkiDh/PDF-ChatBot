from langchain_unstructured import UnstructuredLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

def load_document(doc_path):
    """Load a document from the specified path."""
    try:
        loader = UnstructuredLoader(doc_path)
        documents = loader.load()
        return documents
    except Exception as e:
        raise Exception(f"Error loading document: {str(e)}")

def split_document(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def generate_embeddings():
    """Generate embeddings using HuggingFace model."""
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return embeddings
    except Exception as e:
        raise Exception(f"Error generating embeddings: {str(e)}")

def store_in_chromadb(chunks, embeddings, persist_directory):
    """Store document chunks in ChromaDB with filtered metadata."""
    try:
        # Filter complex metadata
        filtered_chunks = filter_complex_metadata(chunks)
        Chroma.from_documents(
            documents=filtered_chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
    except Exception as e:
        raise Exception(f"Error storing in ChromaDB: {str(e)}")