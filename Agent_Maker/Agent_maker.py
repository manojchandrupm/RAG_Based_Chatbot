from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory

class AgentManager:
    def __init__(self, llm, prompt_template, tools, session_store):
        self.llm = llm
        self.prompt_template = prompt_template
        self.tools = tools
        self.session_store = session_store
        self.agent = self._create_agent(self.tools)
        self.agent_executor = AgentExecutor(
            agent=self.agent, tools=self.tools, verbose=True, handle_parsing_errors=True
        )
        self.agent_with_history = RunnableWithMessageHistory(
            self.agent_executor,
            self.session_store,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

    def _create_agent(self, tools):
        return create_react_agent(
            llm=self.llm,
            prompt=self.prompt_template,
            tools=tools
        )

    def create_subagent(self, subtools, name):
        agent = self._create_agent(subtools)
        agent_executor = AgentExecutor(
            agent=agent, tools=subtools, verbose=True, handle_parsing_errors=True
        )
        agent_with_history = RunnableWithMessageHistory(
            agent_executor,
            self.session_store,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
        return SubAgent(
            name=name,
            agent=agent,
            agent_executor=agent_executor,
            agent_with_history=agent_with_history,
        )

class SubAgent:
    def __init__(self, name, agent, agent_executor, agent_with_history):
        self.name = name
        self.agent = agent
        self.agent_executor = agent_executor
        self.agent_with_history = agent_with_history

    def invoke(self, enriched_query, session_id):
        return self.agent_with_history.invoke(
            {"input": enriched_query},
            config={"configurable": {"session_id": session_id}}
        )