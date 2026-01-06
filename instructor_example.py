# Instructor is a library that can make LLM return specified python type.
# Currently limited to openAI model and vllm, but AWS bedrock can work.

import instructor
from openai import OpenAI
from pydantic import BaseModel

client = instructor.from_openai(OpenAI())
prompt = "Alice is 29 years old"

class User(BaseModel):
    name: str
    age: int

user = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=User,
    messages=[{"role": "user", "content":prompt }],
)

print(user)
