# Setting up browsing web capable agent.
# https://github.com/stanford-mast/blast

'''
1. pip install blastai
2. Run, blastai serve, in terminal.
'''

from openai import OpenAI

# You need to setup interactive LLM in background like Dash, or Flask.
# Use fastAPI to setup LLM like AWS bedrock and get base URL. 
# Or, you could also use vllm if using local models.
base_url = "http://127.0.0.1:8000"

client = OpenAI(api_key="not-needed", base_url=base_url) # Provide api_key if using openAI mmodel. 

stream = client.responses.create(
    model="not-needed",
    input="Compare fried chicken reviews for top 10 fast food restaurants",
    stream=True
)

for event in stream:
    if event.type == "response.output_text.delta":
        print(event.delta if " " in event.delta else "<screenshot>", end="", flush=True)


#===== without using FastAPI(not recommended) ============
from blastai import Engine
import boto3

engine = await Engine.create()

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

def call_bedrock(prompt):
    response = bedrock.invoke_model(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        body=...
    )
    return result

result = await engine.run(
    prompt="Search something",
    llm_fn=call_bedrock
)
