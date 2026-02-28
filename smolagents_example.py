# Local CodeInterpreter using smolagents
# https://medium.com/@laurentkubaski/smolagents-high-level-overview-6b6dea33f9e4

from smolagents import CodeAgent, LiteLLMModel, LogLevel, WebSearchTool

model = LiteLLMModel(
    model_id="openrouter/openai/gpt-4o-mini",
    api_base="https://openrouter.ai/api/v1",
    api_key="[YOUR_API_KEY",
)

prompt = "How many seconds would it take for a leopard at full speed to run through Pont des Arts? "
instructions = "You are not allowed to use internal knowledge for facts. Every factual statement must come from a tool call"
tools = [WebSearchTool()]

agent = CodeAgent(
    model=model,
    tools=tools,
    instructions=instructions,
    verbosity_level=LogLevel.DEBUG,
    max_steps=10,
    use_structured_outputs_internally=True
)

result = agent.run(task=prompt)
print("Result:", result) # 9.62
agent.replay(detailed=False)