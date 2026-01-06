# Force LLM to output specific python types.  Alternatives are instructor, guidance, marvin.
# pydantic ai can work with openAI, local, bedrock, ... models.

from pydantic import BaseModel
from pydantic_ai import Agent

class City(BaseModel):
    name: str
    country: str
    population: int

agent = Agent(
    "openai:gpt-5-mini",
    output_type=list[City],
)

result = agent.run_sync("List the 3 largest cities in Japan")
print(result.output)
