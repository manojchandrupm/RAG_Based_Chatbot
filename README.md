# RAG-Based Samsung Product Manual Chatbot

## Overview
A Retrieval-Augmented Generation chatbot that answers user queries based on Samsung appliance manuals. Uses LangChain, OpenAI LLM/embeddings, and FAISS vector database for scalable semantic search and context-aware responses.

---

## Features
- Multi-PDF document loading and chunking  
- Vector-based semantic search with FAISS  
- OpenAI embeddings and chat completion (GPT)  
- ReAct agent orchestration with custom retriever tools per product  
- Session-aware chat history management  
- Robust prompt engineering and error handling  

---

## Installation

```
git clone https://github.com/yourusername/samsung-rag-chatbot.git
cd RAG_Based_Chatbot
pip install -r requirements.txt
```

**Dependencies:**  
- Python 3.10+  
- `langchain`  
- `langchain_community`  
- `langchain_openai`  
- `faiss-cpu`  
- `openai`  

> **Tip:** If you switch to Qdrant instead of FAISS, also install `langchain-qdrant`.

---

## Setup

1. Place your Samsung product manuals (PDFs) in the `data/` directory.
2. Set your OpenAI API key in your environment:

   ```
   export OPENAI_API_KEY="your-openai-key"
   ```

3. Adjust the list of input PDFs in `trail_11_RAG_chatbot.py` if necessary.

---

## Usage

```
python trail_10_RAG_chatbot.py
```

- Enter product support questions (e.g. "How do I use Power Freeze on my fridge?").
- Type `'q'` or `'quit'` to exit.

---

## Project Structure

- `trail_10_RAG_chatbot.py` — Main chatbot script
- `agent_tools/` — Custom retriever tools (fridge, TV, washing machine)
- `data/` — PDF manuals
- `indexes/` — Vectorstore index files

---

## Extending the Project

- Add new retriever tools for other products.
- Replace FAISS with Qdrant for distributed, cloud-native semantic search.
- Expand prompt logic or session management for richer chat experiences.

---

## License

This project is provided for educational and prototyping purposes.  
Please see LICENSE file or contact for commercial usage details.
```
