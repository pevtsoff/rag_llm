import os
import sys
import chainlit as cl
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM

NUMBER_OF_RETURNED_DOCS = 30
SEARCH_EXTENSIONS = {".py", ".txt", ".sh"}

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    encode_kwargs={"normalize_embeddings": True}
)

def load_and_split_documents(folder_path: str) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
    documents = []

    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if not any(filename.endswith(ext) for ext in SEARCH_EXTENSIONS):
                continue

            filepath = os.path.join(root, filename)
            relative_path = os.path.relpath(filepath, folder_path)

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception:
                continue

            chunks = text_splitter.split_text(content)
            for chunk in chunks:
                documents.append(Document(page_content=chunk, metadata={"source": relative_path}))

    return documents

def show_docs_scores(vectorstore, query) -> str:
    output = []
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=NUMBER_OF_RETURNED_DOCS)
    for idx, (doc, score) in enumerate(docs_and_scores, start=1):
        output.append(f"Score: {score:.4f}\nSource: {doc.metadata['source']}\n{'-' * 40}")
    return "\n".join(output)

def load_faiss_store(folder_path: str, embedding_model) -> VectorStore:
    faiss_index_path = os.path.join(folder_path, ".faiss_store")

    if os.path.exists(faiss_index_path):
        vectorstore = FAISS.load_local(faiss_index_path, embeddings=embedding_model,
                                       allow_dangerous_deserialization=True)
    else:
        documents = load_and_split_documents(folder_path)
        vectorstore = FAISS.from_documents(documents, embedding_model)
        vectorstore.save_local(faiss_index_path)

    return vectorstore

def main(folder_path: str, llm_query: str):
    vectorstore = load_faiss_store(folder_path, embedding_model)
    docs_scores = show_docs_scores(vectorstore, llm_query)

    llm = OllamaLLM(model="deepseek-coder-v2:16b")
    retriever = vectorstore.as_retriever(search_kwargs={"k": NUMBER_OF_RETURNED_DOCS})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    response = qa_chain.invoke(llm_query)
    return docs_scores, response["result"]

# CLI run
if __name__ == "__main__":
    folder_path = sys.argv[1] if len(sys.argv) > 1 else "/home/ivan/ML/monetisation-service/"
    llm_query = sys.argv[2] if len(sys.argv) > 2 else "Can you list me the shell files which create sqs queue in the project"

    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print(f"Invalid folder path: {folder_path}")
        sys.exit(1)

    docs_scores, llm_response = main(folder_path, llm_query)
    print("Top matching documents:\n", docs_scores)
    print("\nLLM Response:\n", llm_response)

# Chainlit UI handlers

# To keep track of state (folder_path), use a simple in-memory variable:
state = {}

@cl.on_chat_start
async def start():
    await cl.Message(content="Please enter the **folder path** to your code directory:").send()

@cl.on_message
async def handle_message(message: cl.Message):
    if "folder_path" not in state:
        folder_path = message.content.strip()
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            await cl.Message(content=f"❌ Invalid path: `{folder_path}`. Please enter a valid folder path:").send()
            return
        state["folder_path"] = folder_path
        await cl.Message(content=f"✅ Folder path set to `{folder_path}`.\n\nNow please enter your **query** for the LLM:").send()
    else:
        llm_query = message.content.strip()
        folder_path = state["folder_path"]

        await cl.Message(content="🔍 Building or loading FAISS index...").send()
        docs_scores, llm_response = main(folder_path, llm_query)

        await cl.Message(content=f"📄 Top matching documents:\n```\n{docs_scores}\n```").send()
        await cl.Message(content=f"💡 LLM Response:\n```\n{llm_response}\n```").send()

        # Reset state for next interaction
        state.clear()
