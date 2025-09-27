from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import os

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from agent_tools.policy_tool import PolicyRetriever

# ____________________________________________________________________________________________________
# | improved version of trail 8 here we used only ChatMessageHistory and RunnableWithMessageHistory  |
# |__________________________________________________________________________________________________|

api_key = os.getenv("OPENAI_API_KEY")

# Load and split document
loader = PyPDFLoader("../data/Policy Clause New India mediclaim  .pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=55)
chunks = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
INDEX_PATH = "../indexes/my_faiss_index"

if os.path.exists(INDEX_PATH):
    vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_PATH)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Enhanced prompt template for clear, concise answers
insurance_react_prompt = PromptTemplate.from_template("""
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

You have access to the following tools:
{tools}

Format:

Question: the input question
Thought: think about what to do next
Action: the action to take, one of [{tool_names}]
Action Input: concise keywords or phrases related to the question
Observation: the result of the action
... (this can repeat)

Thought: I now have enough information
Final Answer: provide a concise, clear answer based solely on observations.

Begin!

{chat_history}

Question: {input}
{agent_scratchpad}
""")


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=api_key)

store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

policy_tool = PolicyRetriever(retriever=retriever, llm=llm)

tools = [policy_tool]

agent = create_react_agent(
    llm=llm,
    prompt=insurance_react_prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,handle_parsing_errors=True)

agent_with_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

session_id = "user-session-1"
# Interaction loop
while True:
    print("Enter your question or 'q' to quit:")
    user_input = input("YOU: ")
    if user_input.lower() in ['q', 'quit', 'exit']:
        print("Goodbye!")
        break
    # Pass concise keywords as Action Input in internal calls for better retrieval
    result = agent_with_history.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )
    print("\nBOT:", result["output"])
    print("-" * 50)
