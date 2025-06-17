import os
import torch
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from pathlib import Path

def load_document(doc_path):
    loader = UnstructuredFileLoader(doc_path)
    documents = loader.load()
    return documents

def split_document(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def generate_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device}
    )
    return embeddings

def store_in_chromadb(chunks, embeddings, persist_directory):
    os.makedirs(persist_directory, exist_ok=True)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vector_store

def main():
    doc_path = input("Please enter the path to your PDF file: ").strip()
    pdf_filename = os.path.basename(doc_path)
    pdf_name = os.path.splitext(pdf_filename)[0]
    persist_directory = f"./chroma_db_{pdf_name}"
    
    if not Path(doc_path).exists():
        print(f"Document {doc_path} not found!")
        return
    if not doc_path.lower().endswith('.pdf'):
        print("Please provide a valid PDF file!")
        return
    
    print("Loading document...")
    documents = load_document(doc_path)
    
    print("Splitting document into chunks...")
    chunks = split_document(documents)
    print(f"Generated {len(chunks)} chunks")
    
    print("Generating embeddings...")
    embeddings = generate_embeddings()
    
    print("Storing embeddings in ChromaDB...")
    vector_store = store_in_chromadb(chunks, embeddings, persist_directory)
    print("Embeddings stored successfully!")

if __name__ == "__main__":
    main()