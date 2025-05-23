import os
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM

# Main Params
FOLDER_PATH = "/home/ivan/ML/monetisation-service/"
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    encode_kwargs={"normalize_embeddings": True}
)
NUMBER_OF_RETURNED_DOCS=30
LLM_QUERY="Can you list me the shell files which create sqs queue in the project"
SEARCH_EXTENSIONS = {".py", ".txt", ".sh"}

# Main Code
def load_and_split_documents(folder_path: str) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
    documents = []
    allowed_extensions = SEARCH_EXTENSIONS

    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if not any(filename.endswith(ext) for ext in allowed_extensions):
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


def show_docs_scores(vectorstore, query):
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=NUMBER_OF_RETURNED_DOCS)
    for idx, (doc, score) in enumerate(docs_and_scores, start=1):
        print(f"Score: {score:.4f}")
        print(f"Document #{idx}: {doc.page_content}")
        print("-" * 40)


def load_faiss_store(embedding_model) -> VectorStore:
    faiss_index_path = "faiss_store"

    if os.path.exists(faiss_index_path):
        print("Loading FAISS index from disk...")
        vectorstore = FAISS.load_local(faiss_index_path, embeddings=embedding_model,
                                       allow_dangerous_deserialization=True)
    else:
        print("Creating new FAISS index and saving to disk...")
        documents = load_and_split_documents(FOLDER_PATH)
        vectorstore = FAISS.from_documents(documents, embedding_model)
        vectorstore.save_local(faiss_index_path)

    return vectorstore

def main():
    vectorstore = load_faiss_store(embedding_model)
    show_docs_scores(vectorstore, LLM_QUERY)

    llm = OllamaLLM(
        model="deepseek-coder-v2:16b"
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": NUMBER_OF_RETURNED_DOCS})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    response = qa_chain.invoke(LLM_QUERY)
    print("LLM Response:", response)


if __name__ == "__main__":
    main()