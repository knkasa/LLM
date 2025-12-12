# Agno to create various agents(multi).

from agno.agent import Agent

agent = Agent(
    model="gpt-4o-mini",
    system_prompt="You are a helpful assistant."
    )

response = agent.run("Summarize the Japan earthquake news today.")
print(response)

# Add Tools (functions) to agents
from agno.tools import tool

@tool
def add(a: int, b: int) -> int:
    return a + b

agent = Agent(
    model="gpt-4o",
    tools=[add]
    )

print(agent.run("Calculate 5 + 20"))

# Use RAG (Faiss, ElasticSearch, ...)
from agno.vectors import MemoryVectorStore

store = MemoryVectorStore()
agent = Agent(model="gpt-4o-mini", vector_store=store)

agent.add_documents([
    "SARIMAX is a time-series forecasting model used by economists.",
    "LSTM networks can capture long-term temporal dependencies."
    ])
agent.run("What are differences between LSTM and SARIMAX.")

# Build Multi-Agent Workflows
from agno.agent import TeamAgent, Agent

researcher = Agent(model="gpt-4o-mini", system_prompt="Do research.")
writer = Agent(model="gpt-4o", system_prompt="Write reports.")

team = TeamAgent(agents=[researcher, writer])
team.run("Write a report about EV adoption in Japan.")

# Create Event-driven / Streaming Agents
for chunk in agent.run("Explain PCA step by step", stream=True):
    print(chunk, end="")

# Agno generating a chart locally
from agno.agent import Agent
from agno.toolkits import VisualizationTools

agent = Agent(
    model="gpt-4o-mini",
    toolkits=[VisualizationTools(output_dir="charts")]
    )

response = agent.run(
    "Create a bar chart with items A=3, B=9, C=5 and save it."
    )
print(response)

