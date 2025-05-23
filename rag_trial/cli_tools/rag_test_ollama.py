from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain.schema import Document

# 1. Документы с полезным контекстом
documents = [
    Document(page_content="Mgzavrebi is a famous pop singer group in Georgia, they sing in ethno pop style"),
    Document(page_content="Mgzavrebi is very famous in Georgia"),
    Document(page_content="Mgzavrebi pretty often give concerts in Tbilisi"),
]
# 2. Эмбеддинги
#embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    encode_kwargs={"normalize_embeddings": True}  # Recommended for BGE
)
# 3. FAISS индекс
vectorstore = FAISS.from_documents(documents, embedding_model)

# 4. LLM через Ollama
llm = OllamaLLM(model="llama3.2:3b")

# 5. Retrieval QA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

# 6. Запрос
query = "What you know about Mgzavrebi?"
response = qa_chain.invoke(query)

# 7. Ответ
print("Ответ:", response)
