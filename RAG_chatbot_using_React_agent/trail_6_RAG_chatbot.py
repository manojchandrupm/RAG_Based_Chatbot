from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import os
from agent_tools.policy_tool import PolicyRetriever

# ________________________________________________________________________________
# | Updated with Manual Chat history , it will just concatenate the previous user|
# | query and answer as a string and pass it to the prompt as {chat history}     |
# |______________________________________________________________________________|

api_key = os.getenv("OPENAI_API_KEY")

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

Conversation History:
{history}

Question: {input}
{agent_scratchpad}
""")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=api_key)

policy_tool = PolicyRetriever(retriever=retriever, llm=llm)

tools = [policy_tool]


agent = create_react_agent(
    llm=llm,
    prompt=insurance_react_prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools,verbose=True)

# Interaction loop
chat_history = ""

while True:
    print("Enter your question or 'q' to quit:")
    user_input = input("YOU: ")
    if user_input.lower() in ['q', 'quit', 'exit']:
        print("Goodbye!")
        break
    chat_history += f"User: {user_input}\n"

    result = agent_executor.invoke({"input": user_input,"history": chat_history})

    answer = result["output"]

    print("\nBOT:", answer)

    print("-" * 50)

    chat_history += f"Bot: {answer}\n"
