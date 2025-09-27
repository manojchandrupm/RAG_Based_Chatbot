import os
import json
from typing import TypedDict, List, Annotated
from operator import itemgetter
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage
from langchain.agents import create_tool_calling_agent
from langgraph.prebuilt import ToolExecutor
from langgraph.graph import StateGraph, END
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain.prompts import PromptTemplate, MessagesPlaceholder

# --- 1. SETUP: Identical to before ---
openai_api_key = os.getenv("OPENAI_API_KEY")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = "https://6f973fc5-fbc1-4866-9aa0-0d28bfe66ffc.eu-west-1-0.aws.cloud.qdrant.io"

qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
embedding_fn = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
vectorstore = QdrantVectorStore(client=qdrant_client, collection_name="chatbot_collection", embedding=embedding_fn)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)

from agent_tools.Pexip_administrator_guide_tool import Pexip_Administrator_Guide_Retriever

pexip_tool = Pexip_Administrator_Guide_Retriever(retriever=retriever,llm=llm)
tools = [pexip_tool]
tool_executor = ToolExecutor(tools)

# --- 2. PROMPTS: Slightly adjusted for a new agent type ---
# This is a more modern prompt for function-calling agents
react_agent_prompt = PromptTemplate.from_template("""
### Role & Persona (Same as your original prompt)
- Primary Function: You are a charismatic and enthusiastic application support assistant for the Pexip Administration Guide.
- Persona: Friendly, patient, and conversational.
- Constraints: Provide answers strictly based on the provided tool's content. If you cannot answer, respond warmly: "I’m sorry, I don’t have that info. Please contact support@[example.com] for help."

Begin!
""")

# Intent clarification prompt remains the same
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


# --- 3. GRAPH DEFINITION (Fully LangGraph Native) ---

# Define the state for our graph. It will hold all necessary information.
class GraphState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    agent_outcome: dict | None  # The decision from the agent (tool call or final answer)
    intermediate_steps: Annotated[list[tuple], itemgetter("intermediate_steps")]  # Tool outputs
    clarification_question: str | None
    final_answer: str | None  # To store the final generated answer


# Define the nodes for the agent's reasoning loop
def run_agent_node(state: GraphState):
    """Decides whether to call a tool or finish."""
    print("---NODE: RUN AGENT---")
    # We use a function-calling agent which is easier to parse in LangGraph
    prompt = PromptTemplate.from_messages([
        ("system", react_agent_prompt.template),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent_runnable = create_tool_calling_agent(llm, tools, prompt)
    agent_outcome = agent_runnable.invoke({
        "input": state["input"],
        "chat_history": state["chat_history"],
        "agent_scratchpad": state["intermediate_steps"]
    })
    return {"agent_outcome": agent_outcome}


def execute_tools_node(state: GraphState):
    """Executes the tools chosen by the agent."""
    print("---NODE: EXECUTE TOOLS---")
    tool_calls = state["agent_outcome"].tool_calls
    messages = []
    for tool_call in tool_calls:
        output = tool_executor.invoke(tool_call)
        messages.append(
            FunctionMessage(content=str(output), name=tool_call["name"], tool_call_id=tool_call["id"])
        )
    return {"intermediate_steps": messages}


# Define the nodes for the initial intent clarification
def clarify_intent_node(state: GraphState):
    """Clarifies the user's intent before handing off to the main agent."""
    print("---NODE: CLARIFY INTENT---")
    history_text = "\n".join([f"{msg.type}: {msg.content}" for msg in state['chat_history']])
    prompt = intent_agent_prompt.format(input=state['input'], history=history_text)
    response_str = llm.invoke(prompt).content
    print(f"Intent Agent Raw Response: {response_str}")
    response_json = json.loads(response_str)

    if response_json.get("status") == "complete":
        print("✅ Intent is complete.")
        # Clear any previous intermediate steps for the new query
        return {"input": response_json["query"], "intermediate_steps": []}
    else:
        print("❓ Intent is incomplete.")
        return {"clarification_question": response_json["clarification_question"]}


# Define conditional logic for routing
def should_route(state: GraphState):
    """Determines where to go after each step."""
    if state.get("clarification_question"):
        return "ask_clarification"  # End the graph to ask the user

    # If there was an agent outcome, decide what to do with it
    if state.get("agent_outcome"):
        if hasattr(state["agent_outcome"], "tool_calls") and state["agent_outcome"].tool_calls:
            return "execute_tools"  # The agent wants to use a tool
        else:
            return "end_conversation"  # The agent has a final answer

    # This is the initial routing from intent clarification
    return "run_agent"


# Build the graph
workflow = StateGraph(GraphState)

workflow.add_node("clarify_intent", clarify_intent_node)
workflow.add_node("run_agent", run_agent_node)
workflow.add_node("execute_tools", execute_tools_node)

workflow.set_entry_point("clarify_intent")

workflow.add_conditional_edges(
    "clarify_intent",
    should_route,
    {
        "ask_clarification": END,  # Stop and wait for user input
        "run_agent": "run_agent",
    },
)

workflow.add_conditional_edges(
    "run_agent",
    should_route,
    {
        "execute_tools": "execute_tools",
        "end_conversation": END,
    }
)
workflow.add_edge("execute_tools", "run_agent")  # Loop back to the agent after tool execution

app = workflow.compile()

# --- 4. INTERACTION LOOP (Slightly modified to handle the new state) ---
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = []
    return store[session_id]


session_id = "user-session-1"
print("Welcome to the Pexip Support Assistant! How can I help you today?")
while True:
    try:
        user_input = input("YOU: ")
        if user_input.lower() in ['q', 'quit', 'exit']:
            print("Goodbye!")
            break

        chat_history = get_session_history(session_id)

        final_state = app.invoke(
            {"input": user_input, "chat_history": chat_history, "intermediate_steps": []},
            config={"configurable": {"session_id": session_id}}
        )

        if final_state.get('clarification_question'):
            print(f"BOT (Clarification): {final_state['clarification_question']}")

        elif final_state.get('agent_outcome'):
            # The final answer is in the 'return_values' of the agent outcome
            final_answer = final_state['agent_outcome'].return_values.get('output', "Sorry, something went wrong.")
            print(f"\nBOT: {final_answer}")
            print("-" * 50)

            # Update history with the user's input and the bot's final answer
            chat_history.extend([
                HumanMessage(content=user_input),
                BaseMessage(type="ai", content=final_answer)
            ])
        else:
            print("Sorry, I encountered an error. Please try again.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        break