import os
import sys
import chainlit as cl
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import create_retrieval_chain, create_history_aware_retriever

NUMBER_OF_RETURNED_DOCS = 30
SEARCH_EXTENSIONS = {".py", ".txt", ".sh"}
MAX_OUTPUT_TOKENS = 50

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


def main(folder_path: str, llm_query: str, chat_history: list[dict]):
    # Load your vectorstore
    vectorstore = load_faiss_store(folder_path, embedding_model)

    # Basic retriever from vectorstore, top-k docs
    base_retriever = vectorstore.as_retriever(k=NUMBER_OF_RETURNED_DOCS)

    # Prepare your LLM instance
    llm = OllamaLLM(model="deepseek-r1:7b", max_tokens=MAX_OUTPUT_TOKENS)

    # Define the prompt for the history-aware retriever
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question, "
                   "which might reference context in the chat history, "
                   "rephrase the question to be a standalone question."),
        ("user", "Chat History:\n{chat_history}\n\nQuestion: {input}")
    ])

    # Create a retriever that is aware of conversation history
    retriever_with_history = create_history_aware_retriever(
        llm,
        base_retriever,
        contextualize_q_prompt
    )

    # Define the QA prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful software developer assistant. "
                   "Answer the question based only on the following context:\n"
                   "{context}\n\n"
                   "If you don't know the answer, say you don't know. "
                   "Keep your answer concise and to the point."),
        ("user", "{input}")
    ])

    # Create a retrieval chain with the history-aware retriever
    retrieval_chain = create_retrieval_chain(
        retriever_with_history,
        qa_prompt | llm
    )

    # Format chat_history for prompt
    formatted_history = "\n".join([
        f"User: {item['query']}\nAssistant: {item['response']}"
        for item in chat_history
        if "query" in item and "response" in item
    ])

    # Run the chain with query and formatted chat history
    result = retrieval_chain.invoke({
        "input": llm_query,
        "chat_history": formatted_history
    })

    # Update chat history
    updated_history = chat_history.copy()
    updated_history.append({"query": llm_query, "response": result["answer"]})

    return updated_history, result["answer"]


# CLI run
if __name__ == "__main__":
    folder_path = sys.argv[1] if len(sys.argv) > 1 else "/home/ivan/ML/monetisation-service/"
    llm_query = sys.argv[2] if len(
        sys.argv) > 2 else "Can you list me the shell files which create sqs queue in the project"

    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print(f"Invalid folder path: {folder_path}")
        sys.exit(1)

    docs_scores, llm_response = main(folder_path, llm_query, [])
    print("Top matching documents:\n", docs_scores)
    print("\nLLM Response:\n", llm_response)


# Chainlit UI handlers
@cl.on_chat_start
async def start():
    cl.user_session.set("chat_history", [])  # Initialize empty chat history list
    cl.user_session.set("folder_path", None)
    await cl.Message(content="Please enter the **folder path** to your code directory:").send()


@cl.on_message
async def handle_message(message: cl.Message):
    # Retrieve conversation history (list of messages) from session
    chat_history = cl.user_session.get("chat_history") or []
    folder_path = cl.user_session.get("folder_path")

    if folder_path is None:
        folder_path_candidate = message.content.strip()
        if not os.path.exists(folder_path_candidate) or not os.path.isdir(folder_path_candidate):
            await cl.Message(
                content=f"❌ Invalid path: `{folder_path_candidate}`. Please enter a valid folder path:").send()
            return
        cl.user_session.set("folder_path", folder_path_candidate)
        await cl.Message(
            content=f"✅ Folder path set to `{folder_path_candidate}`.\n\nNow please enter your **query** for the LLM:").send()
        return

    llm_query = message.content.strip()

    # Show loading message
    msg = cl.Message(content="🔍 Processing your query...")
    await msg.send()

    try:
        # Pass history in, get updated history and answer
        updated_history, llm_response = main(folder_path, llm_query, chat_history)

        # Save updated history back to session
        cl.user_session.set("chat_history", updated_history)

        # Update the loading message with the response
        msg.content = f"💡 LLM Response:\n```\n{llm_response}\n```"
        await msg.update()

    except Exception as e:
        await cl.Message(content=f"❌ An error occurred: {str(e)}").send()