from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate

# Note MCP server needs to be running in background. Create MCP server of your own.
# python mcp_server.py, or
# docker run -d -p 3333:3333 my-mcp-server
client = MultiServerMCPClient(
    servers={
        "docs": {
            "url": "http://localhost:3333",
        },
        "users": {
            "url": "http://localhost:4444",
        },
    }
)

tools = client.get_tools()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
])

agent = create_openai_tools_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)

response = executor.invoke({
    "input": "Find documentation about LangChain memory"
})
print(response["output"])

#Calling MCP tools directly (no agent)
tool = tools[0]
result = tool.invoke({"query": "LangChain memory"})
print(result)
