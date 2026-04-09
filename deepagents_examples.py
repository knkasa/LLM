from deepagents import Agent, tool, Memory
# Large output texts are automatically offloaded to virtual files (if context window exceeds its limit)
# But you can also save it in local.

#========= Tool agent =================================
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from deepagents import create_deep_agent

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny, 22°C in {city}"

agent = create_deep_agent(
    model=init_chat_model("openai:gpt-4o"),
    tools=[get_weather],
    system_prompt="You are a helpful travel assistant.",
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather like in Tokyo?"}]
})

#========= Multi-step reasoning =======================
agent = Agent(
    model="gpt-4o-mini",
    max_steps=5,   # allows multi-step reasoning
    system_prompt="Solve problems step by step."
)

response = agent.run("Find the population of Japan and calculate 10% of it.")


#============ Save conversation in memory =============
memory = Memory()

agent = Agent(
    model="gpt-4o-mini",
    memory=memory,
    system_prompt="Remember user preferences."
)

agent.run("My favorite programming language is Python.")
response = agent.run("What is my favorite language?")

#Save conversation in local. https://pub.towardsai.net/langchain-just-released-deep-agents-and-it-changes-how-you-build-ai-systems-cc2371b04714
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
backend = CompositeBackend(
    routes={"/memories/": StoreBackend(store=store)},
    default=StateBackend(),
)
agent = create_deep_agent(
    tools=[...],
    backend=backend,
    memory=["path/to/AGENTS.md"],  # persistent context file
)

#(another way by claude)
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.middleware.filesystem import FilesystemMiddleware

backend = FilesystemBackend(root_dir="/tmp/agent_context")
agent = create_deep_agent(middleware=[FilesystemMiddleware(backend=backend)])

result = agent.invoke({"messages": [{"role": "user", "content": "Research the history of the internet in detail"}]})

#================ Subagent =====================
from deepagents import create_deep_agent, Subagent

code_reviewer = Subagent(
    name="code-reviewer",
    system_prompt="You are an expert code reviewer. Analyze code for bugs, style, and performance.",
    tools=[read_file_tool],
)
agent = create_deep_agent(
    tools=[internet_search],
    subagents=[code_reviewer],
    system_prompt="You are a research and engineering assistant.",
)

#=============== MCP ==============================
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from deepagents import create_deep_agent
import asyncio

async def run_research_agent():
    async with MultiServerMCPClient({
        "tavily": {
            "command": "npx",
            "args": ["-y", "tavily-mcp"],
            "env": {"TAVILY_API_KEY": "your-key"},
        }
    }) as client:
        tools = client.get_tools()

        agent = create_deep_agent(
            model=init_chat_model("anthropic:claude-sonnet-4-5"),
            tools=tools,
            system_prompt="You are a research assistant. Search and summarize topics thoroughly.",
        )

        result = agent.invoke({
            "messages": [{"role": "user", "content": "What are the latest advancements in quantum computing?"}]
        })
        print(result)

asyncio.run(run_research_agent())