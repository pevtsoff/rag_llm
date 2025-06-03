import os
import sys
import asyncio
import traceback
import chainlit as cl
from httpx import Timeout
from datetime import datetime
from typing import List, Dict, Optional
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import create_retrieval_chain, create_history_aware_retriever

# Constants
OLLAMA_MODEL = "llama3.2:3b"
NUMBER_OF_RETURNED_DOCS = 10
SEARCH_EXTENSIONS = {".py", ".txt", ".sh"}
MAX_OUTPUT_TOKENS = 400
MAX_HISTORY_LENGTH = 20
TIMEOUT_SECONDS = 180.0
FOLDER_PATH = "/home/ivan/ML/monetisation-service/"
QUERY = "Can you list files that create SQS queue"

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5", encode_kwargs={"normalize_embeddings": True}
)


def validate_folder_path(path: str) -> bool:
    """Check if path is a readable directory"""
    return os.path.isdir(path) and os.access(path, os.R_OK)


def sanitize_history(history: List[Dict]) -> List[Dict]:
    """Ensure all history items have correct structure"""
    clean = []
    for msg in history[-MAX_HISTORY_LENGTH:]:  # Truncate first
        if not isinstance(msg, dict):
            continue
        if "query" not in msg or "response" not in msg:
            continue
        clean.append({"query": str(msg["query"]), "response": str(msg["response"])})
    return clean


def load_and_split_documents(folder_path: str) -> List[Document]:
    """Load and split documents from folder"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
    documents = []

    for root, _, files in os.walk(folder_path):
        for filename in files:
            if not any(filename.endswith(ext) for ext in SEARCH_EXTENSIONS):
                continue

            filepath = os.path.join(root, filename)
            relative_path = os.path.relpath(filepath, folder_path)

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                chunks = text_splitter.split_text(content)
                for chunk in chunks:
                    documents.append(
                        Document(page_content=chunk, metadata={"source": relative_path})
                    )
            except Exception as e:
                print(f"Error loading {filepath}: {str(e)}")
                continue

    return documents


def load_faiss_store(folder_path: str, embedding_model) -> VectorStore:
    """Load or create FAISS vector store"""
    faiss_index_path = os.path.join(folder_path, ".faiss_store")

    if os.path.exists(faiss_index_path):
        return FAISS.load_local(
            faiss_index_path,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True,
        )

    documents = load_and_split_documents(folder_path)
    vectorstore = FAISS.from_documents(documents, embedding_model)
    vectorstore.save_local(faiss_index_path)
    return vectorstore


def main(folder_path: str, llm_query: str, chat_history: List[Dict]) -> tuple:
    """Process query with conversation history"""
    # 1. Validate inputs
    if not validate_folder_path(folder_path):
        raise ValueError("Invalid folder path")

    # 2. Load vectorstore
    vectorstore = load_faiss_store(folder_path, embedding_model)
    base_retriever = vectorstore.as_retriever(k=NUMBER_OF_RETURNED_DOCS)

    # 3. Initialize LLM
    llm = OllamaLLM(
        model=OLLAMA_MODEL,
        num_predict=MAX_OUTPUT_TOKENS,
        timeout=Timeout(TIMEOUT_SECONDS),
        temperature=0.3,
    )

    # 4. Setup conversation chain
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Given a chat history and the latest user question, "
                "rephrase it to be a standalone question.",
            ),
            ("user", "History:\n{chat_history}\n\nQuestion: {input}"),
        ]
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful coding assistant. Answer concisely "
                "in 2-3 sentences max using this format:\n"
                "1. <Main point>\n2. <Key detail>\n\nContext:\n{context},"
                "You have a strict limit of 200 tokens. Wrap up your answer clearly."
                "Always give out file names from the context instead of UUIDs",
            ),
            ("user", "{input}"),
        ]
    )

    # 5. Create chains
    retriever_with_history = create_history_aware_retriever(
        llm, base_retriever, contextualize_q_prompt
    )
    retrieval_chain = create_retrieval_chain(retriever_with_history, qa_prompt | llm)

    # 6. Execute
    result = retrieval_chain.invoke(
        {
            "input": llm_query,
            "chat_history": "\n".join(
                f"User: {msg['query']}\nAssistant: {msg['response']}"
                for msg in sanitize_history(chat_history)
            ),
        }
    )

    # 7. Return new history
    new_history = [
        *sanitize_history(chat_history),
        {"query": llm_query, "response": result["answer"]},
    ]
    return new_history, result["answer"]


@cl.on_chat_start
async def init_session():
    """Initialize new chat session"""
    cl.user_session.set("chat_history", [])
    cl.user_session.set("folder_path", None)
    await cl.Message(content="📁 Please enter your project folder path:").send()


@cl.on_message
async def handle_message(message: cl.Message):
    """Handle incoming messages with proper Chainlit API usage"""
    # 1. Get or initialize session state
    chat_history = cl.user_session.get("chat_history", [])
    folder_path = cl.user_session.get("folder_path")

    # 2. Handle folder path setting
    if folder_path is None:
        path_candidate = message.content.strip()
        if not validate_folder_path(path_candidate):
            await cl.Message(content="❌ Invalid path. Please try again:").send()
            return

        cl.user_session.set("folder_path", path_candidate)
        await cl.Message(
            content=f"✅ Analyzing: {path_candidate}\nAsk your question:"
        ).send()
        return

    # 3. Process query - create new message instead of trying to edit
    response_msg = cl.Message(content="")
    await response_msg.send()

    try:
        # 4. Execute with timeout
        new_history, response = await asyncio.wait_for(
            asyncio.to_thread(main, folder_path, message.content, chat_history),
            timeout=TIMEOUT_SECONDS,
        )

        # 5. Update UI using proper Chainlit API
        cl.user_session.set("chat_history", new_history)
        response_msg.content = response
        await response_msg.update()

    except asyncio.TimeoutError:
        response_msg.content = "⏳ Timeout - Please simplify your question"
        await response_msg.update()
    except Exception as e:
        print(f"ERROR: {traceback.format_exc()}")
        response_msg.content = f"⚠️ Error: {str(e)}"
        await response_msg.update()


if __name__ == "__main__":
    # CLI mode for testing
    if len(sys.argv) < 2:
        print("Usage: python script.py <folder_path> [query]")
        sys.exit(1)

    folder_path = sys.argv[1] or FOLDER_PATH
    query = sys.argv[2] if len(sys.argv) > 2 else "What files are in this project?"

    if not validate_folder_path(folder_path):
        print(f"Invalid folder path: {folder_path}")
        sys.exit(1)

    history, response = main(folder_path, query, [])
    print(f"Response: {response}")
