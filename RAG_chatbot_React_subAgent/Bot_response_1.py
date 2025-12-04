import os
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import json

from RAG_chatbot_React_subAgent.agent_tools.Pexip_administrator_guide_tool import Pexip_Administrator_Guide_Retriever
# __________________________________________________________________________________________________
# | - Here i have improved the system and indent agent prompt                                      |
# | - converted the indent response into json format for handling the status of response           |                               |
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
### You are an intent clarification assistant. Your task is to analyze the user's query about the Pexip system and determine if it's complete enough for a support agent to answer.

You must respond ONLY with a single, valid JSON object with one of the following two formats:

1. If the question is clear and actionable:
{{
  "status": "complete",
  "query": "[a clear, rephrased version of the user's question]"
}}

2. If the question is ambiguous or incomplete:
{{
  "status": "incomplete",
  "clarification_question": "[a polite question to ask the user for the missing details]"
}}

### Examples:

User Query: How do I install the Pexip app for Windows?
Your JSON Response:
{{
  "status": "complete",
  "query": "How do I install the Pexip app for Windows?"
}}

---
User Query: My connection...
Your JSON Response:
{{
  "status": "incomplete",
  "clarification_question": "Could you please provide more details about the connection issue you're facing? For example, what happens when you try to connect?"
}}

---
User Query: App doesn’t work
Your JSON Response:
{{
  "status": "incomplete",
  "clarification_question": "I can help with that! Could you please describe the specific problem you're encountering with the Pexip app?"
}}

### Conversation History:
{history}

### User Query:
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

    full_conversation_for_intent = user_input

    while True:
        intent_response_str = get_intent_response(full_conversation_for_intent,session_id)
        print("\nINTENT AGENT:", intent_response_str)
        intent_response = json.loads(intent_response_str)

        if intent_response.get("status") == "complete":
            enriched_query = intent_response["query"]
            print(f"✅ Intent Complete. Passing to main agent: '{enriched_query}'")

            result = agent_with_history.invoke(
                {"input": enriched_query},
                config={"configurable": {"session_id": session_id}}
            )
            print("\nBOT:", result["output"])
            print("-" * 50)
            # Add the complete interaction to the chat history
            get_session_history(session_id).add_user_message(enriched_query)
            get_session_history(session_id).add_ai_message(result["output"])
            break
        elif intent_response.get("status") == "incomplete":
            clarification_question = intent_response["clarification_question"]
            print("BOT (Clarification):", clarification_question)

            follow_up = input("YOU (clarification): ")
            if follow_up.lower() in ['q', 'quit', 'exit']:
                break

            # **KEY IMPROVEMENT**: Combine the original query and the clarification
            full_conversation_for_intent += f"\nAssistant: {clarification_question}\nUser: {follow_up}"
        else:
            print("Sorry, I had a little trouble understanding. Could you please rephrase?")
            break
