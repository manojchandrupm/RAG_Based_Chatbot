import os
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from RAG_chatbot_React_subAgent.agent_tools.Pexip_administrator_guide_tool import Pexip_Administrator_Guide_Retriever
# __________________________________________________________________________________________________
# | - Here we added the qdrant vector DB instead of FAISS                                          |
# | - Also Added the Indent Agent to Clarify the User Input                                        |                               |
# |________________________________________________________________________________________________|
openai_api_key = os.getenv("OPENAI_API_KEY")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = "https://6f973fc5-fbc1-4866-9aa0-0d28bfe66ffc.eu-west-1-0.aws.cloud.qdrant.io"

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key,
)

# Initialize embedding function
embedding_fn = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)

# Connect to existing collection
vectorstore = QdrantVectorStore(
    client=qdrant_client,
    collection_name="chatbot_collection",
    embedding=embedding_fn
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

system_react_prompt = PromptTemplate.from_template("""
### Role
- Primary Function: You are a charismatic and enthusiastic application support assistant dedicated to helping users with the Pexip Administration Guide. Your goal is to provide accurate, clear, and concise answers strictly based on the Pexip Administration Guide manual content.
- Draw upon expert communication principles to craft persuasive, friendly, and engaging responses that build trust.
- Always provide short, digestible responses. Break longer responses into smaller paragraphs or bullet points.

### Persona
- Identity: Friendly, patient, and conversational with a warm, engaging tone.
- Listen attentively to user queries and ask clarifying questions before providing recommendations.
- If asked to address topics outside the manual or product scope, politely inform the user and suggest contacting support at [example@email.com].

### Constraints
1. Never mention access to internal manuals or data explicitly.
2. Keep focus strictly on Pexip Administration Guide support.
3. Gently redirect any off-topic queries back to product support.
4. Provide answers strictly based on the manual content; avoid guessing.
5. If you cannot answer, respond warmly: "I’m sorry, I don’t have that info. Please contact support@[example.com] for help."
6. Use minimal emojis to keep professionalism with friendliness.

You have access to the following tools:
{tools}

Format:

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
# Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")

intent_agent_prompt = PromptTemplate.from_template("""
### You are an intent clarification assistant for questions related to the Pexip system based on the "Infinity Apps Guide for Administrators." Determine if the user's question is complete and clear for the main agent to answer.

Instructions:

If the question is clear, fully detailed, and contains enough information to answer, respond ONLY with:
COMPLETE: [rephrase the user's question clearly]

If the question lacks important details, is ambiguous, or incomplete, respond ONLY with a polite clarifying question to gather more information.

Do NOT ask for clarification if the question contains a subject and a clear intent.

Do NOT add explanations, just respond as "COMPLETE: ..." or a clarifying question.

Examples:

User Query: How do I install the Pexip app for Windows?
Response: COMPLETE: How do I install the Pexip app for Windows?

User Query: I get error code 403 when joining a call.
Response: COMPLETE: I get error code 403 when joining a call in Pexip.

User Query: My connection...
Response: Could you please provide more details about the connection issue you're facing?

User Query: App doesn’t work
Response: Could you please describe the problem you're facing with the Pexip app?

Previous Conversation:
{history}

User Query:
{input}
""")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=openai_api_key)

Pexip_administrator_guide= Pexip_Administrator_Guide_Retriever(retriever=retriever, llm=llm)

tools = [Pexip_administrator_guide]

agent = create_react_agent(
    llm=llm,
    prompt=system_react_prompt,
    tools=tools
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=2)

store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def format_history(chat_history: ChatMessageHistory) -> str:
    # Format recent chat messages to dialogue text
    messages = chat_history.messages[-6:]  # last 6 messages to avoid too long prompt
    formatted = []
    for msg in messages:
        speaker = "User" if msg.type == "human" else "Assistant"
        formatted.append(f"{speaker}: {msg.content}")
    return "\n".join(formatted)

def get_intent_response(user_query: str, session_id: str) -> str:
    chat_history = get_session_history(session_id)
    history_text = format_history(chat_history)

    prompt = intent_agent_prompt.format(input=user_query, history=history_text)
    response = llm.invoke(prompt)
    return response.content

agent_with_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

session_id = "user-session-1"
user_input = ""
while True:
    print("Enter your question or 'q' to quit:")
    user_input = input("YOU: ")
    if user_input.lower() in ['q', 'quit', 'exit']:
        print("Goodbye!")
        break
    user_query = user_input
    while True:
        intent_response = get_intent_response(user_query,session_id)
        print("\nINTENT AGENT:", intent_response)

        if intent_response.startswith("COMPLETE"):
            enriched_query = intent_response.replace("COMPLETE", "").strip()

            result = agent_with_history.invoke(
                {"input": enriched_query},
                config={"configurable": {"session_id": session_id}}
            )
            print("\nBOT:", result["output"])
            print("-" * 50)
            break
        else:
            print("Clarify:", intent_response)
            follow_up = input("YOU (clarification): ")
            if follow_up.lower() in ['q', 'quit', 'exit']:
                break
            else:
                user_input = follow_up

