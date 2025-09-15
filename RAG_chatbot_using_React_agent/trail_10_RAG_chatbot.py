from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.chains import RetrievalQA
import os
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from agent_tools.Bespoke_ai_fridge_tool import BespokeAI_Fridge_Retriever
from agent_tools.Q9FNSeries_TV_tool import Q9FNSeries_TV_Retriever
from agent_tools.washing_machine_tool import WashingMachine_Retriever

# __________________________________________________________________________________________________
# | - Here we added multi document support to get multiple doc as input and store in the vector db |
# | - improved the prompt                                                                          |
# |________________________________________________________________________________________________|

api_key = os.getenv("OPENAI_API_KEY")
INDEX_PATH = "../indexes/my_faiss_index"

pdf_paths = [
    # "data/Policy Clause New India mediclaim  .pdf",
    "data/Samsung_washing_Machine_User_Manual.pdf",
    "data/Samsung_Bespoke_AI_Fridge_user_Manual.pdf",
    "data/Samsung_Q9FNSeries_TV_UserManual.pdf"
]

all_docs = []
for pdf_path in pdf_paths:
    loader = PyPDFLoader(pdf_path)
    all_docs.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=75)

chunks = splitter.split_documents(all_docs)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)

if os.path.exists(INDEX_PATH):
    vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_PATH)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

system_react_prompt = PromptTemplate.from_template("""
You are a product support assistant knowledgeable about Samsung appliances.

You have access to three product manuals in your knowledge base:
- Samsung Bespoke AI Fridge user manual
- Samsung Washing Machine user manual
- Samsung Q9FN Series TV user manual

You are a user manual assistant for three types of Samsung products.
- When a user asks a question, you need to identify the product category by extracting relevant keywords from the user’s question, then provide an answer based on the corresponding product’s manual.
- Once you choose a tool and retrieve chunks using that tool, please do not call the tool repeatedly. Use the retrieved chunks to answer the user's query without making unnecessary additional calls.
- If you cannot find any relevant information in the database, ask the user to specify the product and provide more details about their query.
- Once the relevant document chunks are retrieved for the identified product, do not perform additional retrievals during the same query.
- When you receive user question, identify relevant product and do **one** retrieval using its tool.
- Use only the returned chunks to answer the question.
- Do NOT call the retrieval tool more than once per question.
- If no relevant information is found, respond: "I'm sorry, I do not have that information in the manual."
- Produce a concise, clear final answer referencing the document info.

Rules you must follow:
1. Based on the user's question, first identify which product manual is relevant.
2. Answer strictly using only the information from the identified product manual context.
3. Do not use information from other product manuals or outside knowledge.
4. Present clear and concise answers.
5. Use bullet points or numbered lists if the context includes them.
6. Avoid repeating large text blocks; summarize key points.
7. Preserve formal wording of product features, warnings, and instructions.
8. Once you have located relevant information, stop retrieving further documents.
9. Summarize all findings into a clear, final answer to the user's question.
10. If the relevant information is not found in the identified manual, respond with:
   "I'm sorry, I do not have that information in the  manual."

You have access to the following tools:
{tools}

Format:

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=api_key)

Bespoke_ai_fridge_tool = BespokeAI_Fridge_Retriever(retriever=retriever, llm=llm)
Q9FNSeries_TV_tool = Q9FNSeries_TV_Retriever(retriever=retriever, llm=llm)
washing_machine_tool = WashingMachine_Retriever(retriever=retriever, llm=llm)

tools = [Bespoke_ai_fridge_tool, Q9FNSeries_TV_tool, washing_machine_tool]

agent = create_react_agent(
    llm=llm,
    prompt=system_react_prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,handle_parsing_errors=True)

store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

agent_with_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)
session_id = "user-session-1"

while True:
    print("Enter your question or 'q' to quit:")
    user_input = input("YOU: ")
    if user_input.lower() in ['q', 'quit', 'exit']:
        print("Goodbye!")
        break

    result = agent_with_history.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )
    print("\nBOT:", result["output"])
    print("-" * 50)