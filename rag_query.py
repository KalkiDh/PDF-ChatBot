import os
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_chroma import Chroma
import vector_utils
from dotenv import load_dotenv
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

def initialize_llm():
    load_dotenv()
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError("GITHUB_TOKEN not found in .env file")
    endpoint = "https://models.github.ai/inference"
    model = "xai/grok-3"
    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(token),
    )
    return client

def create_rag_chain(vector_store, llm):
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant that answers questions based on provided context. Use the following context to answer the question concisely in about 100 words:\n\n{context}"),
        ("user", "{question}")
    ])
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    def invoke_llm(inputs):
        messages = [
            SystemMessage(content=inputs.messages[0].content),
            UserMessage(content=inputs.messages[1].content)
        ]
        response = llm.complete(
            messages=messages,
            temperature=0.7,
            top_p=0.9,
            model="xai/grok-3"
        )
        return response.choices[0].message.content
    chain = (
        {
            "context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
            "question": RunnablePassthrough()
        }
        | prompt_template
        | RunnableLambda(invoke_llm)
    )
    return chain

def main():
    doc_path = input("Please enter the path to your PDF file: ").strip().strip('"\'')
    doc_path = os.path.normpath(doc_path)
    
    if not Path(doc_path).exists():
        print(f"Error: Document '{doc_path}' not found!")
        return
    if not doc_path.lower().endswith('.pdf'):
        print(f"Error: File '{doc_path}' is not a PDF.")
        return
    
    pdf_filename = os.path.basename(doc_path)
    pdf_name = os.path.splitext(pdf_filename)[0]
    persist_directory = f"./chroma_db_{pdf_name}"
    
    embeddings = vector_utils.generate_embeddings()
    
    if os.path.exists(persist_directory):
        print(f"Loading existing vector store from {persist_directory}...")
        try:
            vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return
    else:
        print(f"Processing document: {doc_path}...")
        try:
            documents = vector_utils.load_document(doc_path)
            chunks = vector_utils.split_document(documents)
            vector_store = vector_utils.store_in_chromadb(chunks, embeddings, persist_directory)
            print("Document processed and embeddings stored successfully!")
        except Exception as e:
            print(f"Error processing document: {e}")
            return
    
    print("Initializing text generation model...")
    try:
        llm = initialize_llm()
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return
    
    print("Creating RAG chain...")
    chain = create_rag_chain(vector_store, llm)
    
    query = input("Please enter your query about the document: ").strip()
    if not query:
        print("Error: Query cannot be empty!")
        return
    
    print("\nProcessing query...")
    try:
        result = chain.invoke(query)
        print(f"\nQuery: {query}")
        print(f"Response: {result}")
    except Exception as e:
        print(f"Error processing query: {e}")

if __name__ == "__main__":
    main()