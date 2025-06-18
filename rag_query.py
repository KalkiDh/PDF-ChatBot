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
from azure.core.exceptions import HttpResponseError, ServiceRequestError

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
        try:
            self.vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=vector_utils.generate_embeddings()
            )
            logger.info(f"Vector store loaded from {persist_directory}")
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise e

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
            logger.warning("No relevant document chunks retrieved")
            response = "No relevant information found in the document."
            self._update_history(user_query, response)
            logger.info(f"Response time: {time.time()-start_time:.1f}s")
            return response
        
        context = "\n\n".join(doc.page_content for doc in context_docs)
        logger.debug(f"Context sent to model: {context[:200]}...")
        
        # Build message sequence with formatted system prompt
        system_prompt = f"""You are an assistant that answers questions based solely on the provided document context. For summary queries (e.g., "Summarise this file"), provide a detailed, structured response in markdown format with the following template:

# Summary of [Document Name]
[Brief overview of the document]

## [Section 1]
- [Point 1]
- [Point 2]
- [Point 3]

## [Section 2]
- [Point 1]
- [Point 2]
- [Point 3]

## [Section N]
- [Point 1]
- [Point 2]

For other queries, answer concisely in the exact format requested (e.g., 50-word summary). Do not assume or answer a different question.

Context:
{context}"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            *self._format_history(),
            {"role": "user", "content": user_query}
        ]
        
        # Log full prompt
        prompt_log = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        logger.debug(f"Full prompt sent to model:\n{prompt_log}")
        
        # Get LLM response
        try:
            response_obj = self.client.complete(
                messages=messages,
                temperature=0.7,
                top_p=0.9,
                model=self.model,
                timeout=30
            )
            logger.info("Azure API response received successfully")
            logger.debug(f"Raw API response: {response_obj}")
            
            if response_obj.choices and len(response_obj.choices) > 0:
                response_content = response_obj.choices[0].message.content
                if not response_content:
                    logger.warning("Empty response content received")
                    response_content = "No valid response from the model."
            else:
                logger.error("No choices in API response")
                response_content = "Invalid API response format."
                
        except HttpResponseError as e:
            logger.error(f"Azure HTTP error: {str(e)}")
            response_content = f"API error: {str(e)}"
        except ServiceRequestError as e:
            logger.error(f"Azure service request error: {str(e)}")
            response_content = f"Network error: {str(e)}"
        except TimeoutError:
            logger.error("Azure API request timed out")
            response_content = "API request timed out after 30 seconds."
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            response_content = f"Unexpected error: {str(e)}"
        
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
        try:
            docs = vector_utils.load_document(doc_path)
            chunks = vector_utils.split_document(docs)
            vector_utils.store_in_chromadb(
                chunks,
                vector_utils.generate_embeddings(),
                persist_dir
            )
            logger.info(f"Vector store created at {persist_dir}")
        except Exception as e:
            print(f"Error indexing document: {e}")
            logger.error(f"Document indexing error: {str(e)}")
            return
    
    try:
        client = RAGChatClient(persist_dir)
    except Exception as e:
        print(f"Error initializing RAG client: {e}")
        logger.error(f"RAG client initialization error: {str(e)}")
        return
    
    print("Ask questions (type 'exit' to quit):")
    
    while (query := input("\nYou: ").strip().lower()) not in ('exit', 'quit'):
        if not query:
            print("Error: Query cannot be empty!")
            continue
        
        try:
            response = client.get_response(query)
            print(f"\nAssistant: {response}")
        except Exception as e:
            print(f"Error: {str(e)}")
            logger.error(f"Query processing error: {str(e)}")

if __name__ == "__main__":
    main()