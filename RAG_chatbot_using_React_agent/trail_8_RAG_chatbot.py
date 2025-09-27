from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import os

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder,ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# _______________________________________________________________________________
# | - here I updated from manual chat history to framework chat history by using |
# |  ChatMessageHistory,BaseChatMessageHistory,RunnableWithMessageHistory        |
# |______________________________________________________________________________|
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=api_key)

# Load and split document
loader = PyPDFLoader("../data/Policy Clause New India mediclaim  .pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=75)
chunks = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
INDEX_PATH = "../indexes/my_faiss_index"

if os.path.exists(INDEX_PATH):
    vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_PATH)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Enhanced prompt template for clear, concise answers
system_prompt = ("""
You are an insurance policy assistant specializing in answering questions from the 
"New India Mediclaim Policy" document.

Rules you must follow:
1. Answer strictly using only the information in the provided context.
2. Do not use outside knowledge.
3. Present clear and concise answers.
4. Use bullet points or numbered lists if context includes them.
5. Avoid repeating large text blocks; summarize key points.
6. Preserve formal wording of company names, legal clauses, and definitions.
7. Once you have located relevant information, stop retrieving further documents.  
8. Summarize all findings into a clear, final answer to the user's question.

Begin!

answer concise.
    "\n\n""
    "{context}"
""")

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Interaction loop
while True:
    print("Enter your question or 'q' to quit:")
    user_input = input("YOU: ")
    if user_input.lower() in ['q', 'quit', 'exit']:
        print("Goodbye!")
        break
    # Pass concise keywords as Action Input in internal calls for better retrieval
    result = conversational_rag_chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": "abc123"}},
        )["answer"]
    print("\nBOT:", result)
    print("-" * 50)
