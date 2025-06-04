### Description
The goal of this project is to investigate how to create a local AI assistant for search
within the folder of your documents

### How to launch
1. Install ollama
https://ollama.com/download
2. Pull llama3.2:3b model with ollama
```
ollama pull llama3.2:3b
ollama run llama3.2:3b 
```
3. poetry install
4. Update .env file with your params 
5. Launch

**Launch Web UI:**

```
chainlit run rag_trial/cli_tools/rag_code_base.py

In the browsers open chat window - enter the path of your code folder like:
"/home/ivan/ML/monetisation-service/"

Enter the query about your code :)
```

**Launch via CLI:**
```
python rag_trial/cli_tools/rag_code_base.py "code path" "your query"
example:
python rag_trial/cli_tools/rag_code_base.py "/home/ivan/ML/monetisation-service/" "where in the project I save customer into the database?"

```

### Key Params
```
OLLAMA_MODEL = "llama3.2:3b" # the name of the local ollama model
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5" # the name of hugging face embedding model to vectorize documents
NUMBER_OF_RETURNED_DOCS = 10 # Number of document that vector stor puts into the model context. The higher is better
and thus, SLOWER! 
MAX_OUTPUT_TOKENS = 400
MAX_HISTORY_LENGTH = 20
TIMEOUT_SECONDS = 180.0 # Timeout to wait a response from Ollama model
FOLDER_PATH = "/home/ivan/ML/monetisation-service/" # Path to Your Code or document folder
QUERY = "Can you list files that create SQS queue" # CLI query to LLM model
```