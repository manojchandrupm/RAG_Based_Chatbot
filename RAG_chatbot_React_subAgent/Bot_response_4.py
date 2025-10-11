import os
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain.prompts import PromptTemplate, MessagesPlaceholder

from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_community.chat_message_histories import ChatMessageHistory
import json
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory
from langgraph.graph import StateGraph, START, END

# -----------  SETUP  -------------
openai_api_key = os.getenv("OPENAI_API_KEY")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = "https://6f973fc5-fbc1-4866-9aa0-0d28bfe66ffc.eu-west-1-0.aws.cloud.qdrant.io:6333"


qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
embedding_fn = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
vectorstore = QdrantVectorStore(client=qdrant_client, collection_name="chatbot_collection", embedding=embedding_fn)

retriever_pexip = vectorstore.as_retriever(search_kwargs={
    "k": 3,
    "filter": {
        "must": [{
            "key": "document_id",
            "match": {"value": "Infinity_Apps_Guide_for_Administrators_v38"}
        }]
    }
})
retriever_brother = vectorstore.as_retriever(search_kwargs={
    "k": 3,
    "filter": {
        "must": [{
            "key": "document_id",
            "match": {"value": "Brother_software_user_guide"}
        }]
    }
})


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)

# -----------  Tools  -------------
from agent_tools.Pexip_administrator_guide_tool import Pexip_Administrator_Guide_Retriever
from agent_tools.Brother_software_tool import Brother_software_Retriever
pexip_tool = Pexip_Administrator_Guide_Retriever(retriever=retriever_pexip,llm=llm)
brother_tool = Brother_software_Retriever(retriever=retriever_brother,llm=llm)
tools = [pexip_tool, brother_tool]

# -----------  PROMPTS  -------------

react_agent_prompt = PromptTemplate.from_template("""
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

previous chatHistory :
{chat_history}

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


intent_agent_prompt = PromptTemplate.from_template("""
### You are an intent clarification assistant... (Same as your original prompt)

You must respond ONLY with a single, valid JSON object with one of the following two formats:

1. If the question is clear and actionable:
{{
  "status": "complete",
  "query": "[a clear, rephrased version of the user's question]"
}}

2. If the question is incomplete or ambiguous:
{{
  "status": "incomplete",
  "clarification_question": "[a polite question to ask the user for the missing details]"
}}

### Conversation History:
{history}

### User Query:
{input}
""")

store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    raw_user_input: str
    enriched_query: str | None
    clarification_question: str | None
    response: str | None
    session_id: str

def format_history(chat_history: ChatMessageHistory) -> str:
    messages = chat_history.messages[-6:]
    formatted = []
    for msg in messages:
        speaker = "User" if msg.type == "human" else "Assistant"
        formatted.append(f"{speaker}: {msg.content}")
    return "\n".join(formatted)

def intent_agent_node(state: AgentState):
    user_input = state["raw_user_input"]
    session_id = state["session_id"]

    full_conversation_for_intent = user_input
    while True:
        chat_history = get_session_history(session_id)
        history_text = format_history(chat_history)
        prompt = intent_agent_prompt.format(input=full_conversation_for_intent, history=history_text)
        response = llm.invoke(prompt)
        intent_response_str = response.content
        print("\nINTENT AGENT:", intent_response_str)
        intent_response = json.loads(intent_response_str)

        if intent_response.get("status") == "complete":
            enriched_query = intent_response["query"]
            print(f"✅ Intent Complete. Passing to main agent: '{enriched_query}'")
            return {
                **state,
                "enriched_query": enriched_query,
            }
        elif intent_response.get("status") == "incomplete":
            clarification_question = intent_response["clarification_question"]
            print("BOT (Clarification):", clarification_question)

            follow_up = input("YOU (clarification): ")
            if follow_up.lower() in ['q', 'quit', 'exit']:
                break

            full_conversation_for_intent += f"\nAssistant: {clarification_question}\nUser: {follow_up}"

agent = create_react_agent(
    llm=llm,
    prompt=react_agent_prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,handle_parsing_errors=True)

agent_with_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)
# Create per-tool agents/executors
pexip_agent = create_react_agent(
    llm=llm,
    prompt=react_agent_prompt,
    tools=[pexip_tool]
)
pexip_agent_executor = AgentExecutor(agent=pexip_agent, tools=[pexip_tool], verbose=True, handle_parsing_errors=True)
pexip_agent_with_history = RunnableWithMessageHistory(
    pexip_agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

brother_agent = create_react_agent(
    llm=llm,
    prompt=react_agent_prompt,
    tools=[brother_tool]
)
brother_agent_executor = AgentExecutor(agent=brother_agent, tools=[brother_tool], verbose=True, handle_parsing_errors=True)
brother_agent_with_history = RunnableWithMessageHistory(
    brother_agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)


def select_tool_name(enriched_query: str) -> str:
    query_lower = enriched_query.lower()
    if any(keyword in query_lower for keyword in ["pexip", "conference", "administration", "Pexip_Administrator_Guide_Retriever"]):
        return "pexip"
    elif any(keyword in query_lower for keyword in ["brother", "printer" ,"Brother Software User Guide"]):
        return "brother"
    else:
        return "pexip"

def react_agent_node(state: AgentState) -> AgentState:
    enriched_query = state["enriched_query"]
    session_id = state["session_id"]

    tool_name = select_tool_name(enriched_query)
    if tool_name == "pexip":
        result = pexip_agent_with_history.invoke(
            {"input": enriched_query},
            config={"configurable": {"session_id": session_id}}
        )
    elif tool_name == "brother":
        result = brother_agent_with_history.invoke(
            {"input": enriched_query},
            config={"configurable": {"session_id": session_id}}
        )
    else:
        # fallback fallback tool or response
        result = agent_with_history.invoke(
            {"input": enriched_query},
            config={"configurable": {"session_id": session_id}}
        )

    print("\nBOT:", result["output"])
    print("-" * 50)
    # Add the complete interaction to the chat history
    get_session_history(session_id).add_user_message(enriched_query)
    get_session_history(session_id).add_ai_message(result["output"])

    return {
        **state,
        "response": result["output"]
    }

# -------------  GRAPH  -----------------
workflow = StateGraph(AgentState)

workflow.add_node("intent_agent", intent_agent_node)
workflow.add_node("react_agent", react_agent_node)

# Set START input to intent_agent node (user raw input goes here)
workflow.set_entry_point("intent_agent")

# Conditional edge based on intent_agent output
def intent_edge(state: AgentState):
    if state.get("enriched_query"):
        return "react_agent"
    else:
        return "intent_agent"

workflow.add_conditional_edges("intent_agent", intent_edge, {
    "react_agent": "react_agent",
    "intent_agent": "intent_agent",
})

# Add normal edge back from react_agent to START or END graph
workflow.add_edge("react_agent", END)

graph = workflow.compile()

# ------------  INTERACTION LOOP --------------

session_id = "user-session-1"
print("Welcome to the Support Assistant! How can I help you today?")
while True:
    try:
        user_input = input("YOU: ")
        if user_input.lower() in ['q', 'quit', 'exit']:
            print("Goodbye!")
            break
        state = {
            "raw_user_input": user_input,
            "messages": [],
            "enriched_query": None,
            "clarification_question": None,
            "response": None,
            "session_id": session_id
        }
        result_state = graph.invoke(state)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        break