import os
import time
import logging
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_chroma import Chroma
import vector_utils
from dotenv import load_dotenv
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGChatClient:
    def __init__(self, persist_directory):
        load_dotenv()
        self.token = os.getenv("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("Missing GITHUB_TOKEN in .env")
        
        self.client = ChatCompletionsClient(
            endpoint="https://models.github.ai/inference",
            credential=AzureKeyCredential(self.token)
        )
        self.model = "xai/grok-3"
        self.conversation_history = []
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=vector_utils.generate_embeddings()
        )
        logger.info(f"Vector store loaded from {persist_directory}")

    def _retrieve_context(self, query):
        """Retrieve relevant document chunks"""
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query)
        logger.info(f"Retrieved {len(docs)} document chunks for query: {query}")
        return docs

    def get_response(self, user_query):
        start_time = time.time()
        
        # Retrieve context
        context_docs = self._retrieve_context(user_query)
        if not context_docs:
            response = "No relevant information found in the document."
            self._update_history(user_query, response)
            return response
        
        context = "\n\n".join(doc.page_content for doc in context_docs)
        
        # Build message sequence
        messages = [
            {"role": "system", "content": f"You are an assistant that answers questions based solely on the provided document context. Answer the user's query concisely in the exact format requested (e.g., 50-word summary). Do not assume or answer a different question. Context:\n\n{context}"},
            *self._format_history(),
            {"role": "user", "content": user_query}
        ]
        
        # Log full prompt
        prompt_log = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        logger.debug(f"Full prompt sent to model:\n{prompt_log}")
        
        # Get LLM response
        response_obj = self.client.complete(
            messages=messages,
            temperature=0.7,
            top_p=0.9,
            model=self.model
        )
        
        response_content = response_obj.choices[0].message.content
        self._update_history(user_query, response_content)
        
        logger.info(f"Response time: {time.time()-start_time:.1f}s")
        return response_content

    def _format_history(self):
        """Convert LangChain messages to Azure format"""
        return [
            {"role": "assistant" if isinstance(msg, AIMessage) else "user", "content": msg.content}
            for msg in self.conversation_history[-4:]
        ]

    def _update_history(self, user_query, response):
        """Store history in LangChain format"""
        self.conversation_history.extend([
            HumanMessage(content=user_query),
            AIMessage(content=response)
        ])

def main():
    doc_path = input("PDF path: ").strip().strip('"\'')
    doc_path = os.path.normpath(doc_path)
    
    if not Path(doc_path).exists():
        print(f"Error: Document '{doc_path}' not found!")
        return
    if not doc_path.lower().endswith('.pdf'):
        print(f"Error: File '{doc_path}' is not a PDF.")
        return
    
    pdf_name = Path(doc_path).stem
    persist_dir = f"./chroma_db_{pdf_name}"
    
    if not Path(persist_dir).exists():
        print("Indexing document...")
        docs = vector_utils.load_document(doc_path)
        chunks = vector_utils.split_document(docs)
        vector_utils.store_in_chromadb(
            chunks,
            vector_utils.generate_embeddings(),
            persist_dir
        )
        logger.info(f"Vector store created at {persist_dir}")
    
    client = RAGChatClient(persist_dir)
    
    print("Ask questions (type 'exit' to quit):")
    
    while (query := input("\nYou: ").strip().lower()) not in ('exit', 'quit'):
        if not query:
            print("Error: Query cannot be empty!")
            continue
        
        response = client.get_response(query)
        print(f"\nAssistant: {response}")

if __name__ == "__main__":
    main()