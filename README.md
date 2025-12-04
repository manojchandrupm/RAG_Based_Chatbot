# RAG-Based Chatbot with Multi-Document Support 🚀

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot supporting **Pexip Administrator Guide** and **Brother Software User Guide**. Built with **LangChain ReAct agents**, **Qdrant vector database**, **OpenAI GPT-4o-mini**, and **Flask API** with session-aware chat history and intent clarification.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2.16-orange.svg)](https://python.langchain.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Cloud-green.svg)](https://qdrant.tech/)
[![Flask](https://img.shields.io/badge/Flask-API-green.svg)](https://flask.palletsprojects.com/)

## ✨ **Key Upgrades**
- ✅ **Switched from FAISS to cloud Qdrant** for scalable vector search
- ✅ **Intent clarification sub-agent** for ambiguous queries
- ✅ **Session-persistent chat history** with 30-min cleanup
- ✅ **Specialized sub-agents** (Pexip/Brother) with dynamic tool routing
- ✅ **Production Flask API** + responsive HTML chat UI

## 🚀 **Features**
- **Multi-Document RAG**: Separate Qdrant collections for Pexip & Brother manuals
- **Intelligent Summarization**: GPT-4o-mini condenses PDF pages for better retrieval
- **ReAct Agent Architecture**: Primary agent + specialized sub-agents
- **Intent Clarification**: Automatically asks clarifying questions
- **Session Management**: Chat history per user session (in-memory, Redis-ready)
- **REST API**: `/chat` endpoint for easy frontend integration
- **Real-time Chat UI**: Responsive HTML/JS interface

## 🛠️ **Quick Start**

### 1. Clone & Install
git clone <your-repo-url>
cd RAG_Based_Chatbot
pip install -r requirements.txt

### 2. Set Environment
export OPENAI_API_KEY="your-openai-api-key"


### 3. Initialize Vector Store
python cluster_creation.py # Creates Qdrant collections
python Qdrant_store_3.py # Ingests PDF manuals


### 4. Launch Chatbot
python chatbot_api.py


📱 **Open**: [http://localhost:5000](http://localhost:5000)

## 📁 **Project Structure**
├── cluster_creation.py # Qdrant collection setup
├── Qdrant_store_3.py # PDF ingestion + summarization
├── Agent_maker.py # ReAct agent factory + sub-agents
├── chatbot_api.py # Flask API + LangGraph workflow
├── chat.html # Responsive chat UI
├── agent_tools/ # Custom retriever tools
│ ├── Pexip_administrator_guide_tool.py
│ └── Brother_software_tool.py
└── data/ # 📄 PDF manuals go here


## 🎯 **API Endpoints**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` or `/Homepage` | 🎨 Serve chat UI |
| `POST` | `/chat` | 💬 `{"query": "...", "session_id": "user1"}` |
| `GET` | `/test-history` | 📊 View active sessions |

**cURL Example:**
curl -X POST http://localhost:5000/chat
-H "Content-Type: application/json"
-d '{"query": "Brother printer setup", "session_id": "session-123"}'


## 🧠 **How It Works**

User Query ──→ Intent Agent (clarify) ──→ Tool Router ──→ Sub-Agent (Pexip/Brother)
│
↓
Qdrant Retrieval ──→ GPT-4o-mini ──→ Session History ──→ Response


1. **Intent Agent** clarifies ambiguous queries
2. **Tool Router** selects Pexip/Brother sub-agent
3. **Retriever Tool** searches relevant Qdrant collection
4. **ReAct Agent** reasons + generates response
5. **Chat History** maintains context

## ⚡ **Quick Demo**
python chatbot_api.py

http://localhost:5000

Try: "How do I configure Pexip VMR?"
"Brother printer won't connect"



## 🔄 **Extending**

### ➕ Add New Documents
1. Update pdf_paths in Qdrant_store_3.py
2. Run ingestion
python Qdrant_store_3.py

3. Add retriever tool in agent_tools/


### 🚀 Production Deployment
Redis for persistent sessions
pip install redis

Dockerize Flask app
docker build -t rag-chatbot .
docker run -p 5000:5000 rag-chatbot



## 📊 **Tech Stack**

| Component | Technology |
|-----------|------------|
| **Vector DB** | Qdrant Cloud (AWS) |
| **Embeddings** | OpenAI `text-embedding-3-small` |
| **LLM** | OpenAI `gpt-4o-mini` |
| **Agents** | LangChain ReAct + LangGraph |
| **Backend** | Flask REST API |
| **Frontend** | HTML/CSS/JS |

## 📦 **Core Dependencies**
langchain==0.2.16
langchain-openai
langchain-qdrant
qdrant-client
flask
langgraph
openai
pypdf



## 🚀 **Run Order**
cluster_creation.py # Setup Qdrant

Qdrant_store_3.py # Ingest PDFs

chatbot_api.py # Start API + UI


## 📄 **License**
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MIT License - Free for educational/commercial use.

---

**⭐ Built for SkillMate Internship**  
*Multi-agent RAG system with production-ready API and session management*
