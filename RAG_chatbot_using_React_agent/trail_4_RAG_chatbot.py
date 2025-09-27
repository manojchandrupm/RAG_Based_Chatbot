from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import os
from langchain.memory import ConversationSummaryMemory
from agent_tools.policy_tool import PolicyRetriever

# _____________________________________________________________________________
# |Here we added the langchain - ConversationSummaryMemory to add chat history|
# |but it will not support in the new version of langchain .                  |
# |___________________________________________________________________________|

api_key = os.getenv("OPENAI_API_KEY")

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

insurance_react_prompt = PromptTemplate.from_template("""
You are an insurance policy assistant specializing in answering questions from the 
"New India Mediclaim Policy" document.

Rules you must follow:
1. Answer strictly using only the information in the provided context.
2. Do not use outside knowledge, even if you think you know the answer.
3. Present the answer clearly, using bullet points or numbered steps if the context contains lists or procedures.
4. If the context contains formal text (like company name, address, or legal clauses), preserve that wording in the response.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question
Thought: think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input for that action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Begin!

Question: {input}
{agent_scratchpad}
""")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=api_key)

memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="chat_history",  # Used internally by LangChain
    return_messages=True
)

policy_tool = PolicyRetriever(retriever=retriever, llm=llm)

tools=[]
tools.append(policy_tool)

agent = create_react_agent(
    llm = llm,
    prompt = insurance_react_prompt,
    tools = tools
)

agent_executer =AgentExecutor(agent=agent,tools=tools,verbose=True,memory=memory)

while True:
    print("Enter your Question here or enter 'q' to quit:")
    user_input = input("YOU: ")
    if user_input.lower() in ["exit", "quit", "q"]:
        print("Goodbye")
        break
    result = agent_executer.invoke({"input": user_input})
    print("\nBOT:", result["output"])
    print("------------------------------------------------")