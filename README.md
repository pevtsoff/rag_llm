### Description

Simple example of RAG with ollama model

### How to launch
1. Install ollama
https://ollama.com/download
2. Pull llama3.2:3b model
3. ollama run llama3.2:3b
4. pip install -r requirements.txt 
5.
6. Launch

**UI Option:**

```
chainlit run rag_trial/cli_tools/rag_code_base.py

In the browsers open chat window - enter the path of your code folder like:
"/home/ivan/ML/monetisation-service/"

Enter the query about your code :)
```

**CLI Option:**
```
python rag_trial/cli_tools/rag_code_base.py "code path" "your query"
example:
python rag_trial/cli_tools/rag_code_base.py "/home/ivan/ML/monetisation-service/" "where in the project I save customer into the database?"

```
