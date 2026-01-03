import os
import boto3
import json
from bs4 import BeautifulSoup
import dspy
from dspy.teleprompt import BootstrapFewShot

# For Claude model from Bedrock, you likely need to setup keys through environment variable. Use other models like Amazon Nove.
os.environ["AWS_REGION"] = "us-east-1"
os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""

class BedrockClaudeLM(dspy.LM):
    def __init__(
        self,
        model_id: str,
        region="us-east-1",
        temperature=0.5,
        max_tokens=5000,
        aws_access_key_id=None,
        aws_secret_access_key=None,
    ):
        super().__init__(model=model_id)

        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

        self.client = session.client(
            "bedrock-runtime",
            region_name=region
        )

        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens

def __call__(self, prompt=None, messages=None, **kwargs):
    if messages is not None:
        prompt = "\n".join(
            m["content"] if isinstance(m["content"], str)
            else m["content"][0]["text"]
            for m in messages
        )

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "temperature": self.temperature,
        "max_tokens": self.max_tokens,
        "messages": [
            {
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": (  # Make response in json otherwise it will not work.
                        "You must respond ONLY in valid JSON.\n"
                        "The JSON must contain exactly these fields:\n"
                        "- reasoning\n"
                        "- answer\n"
                        "Do not include any extra text outside JSON."
                    )
                }]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ],
    }

    response = self.client.invoke_model(
        modelId=self.model_id,
        body=json.dumps(payload),
        contentType="application/json",
        accept="application/json",
    )

    body = json.loads(response["body"].read())

    # Claude returns text, but now it's JSON.  #TODO: use try-except, or pydantic to force response in json format.
    return body["content"][0]["text"]


bedrock_lm = BedrockClaudeLM(
    model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
    aws_access_key_id="",
    aws_secret_access_key="",
    temperature=0.5,
    )


dspy.settings.configure(lm=bedrock_lm, adapter=dspy.ChatAdapter())
#dspy.settings.configure(lm=bedrock_lm)


class QuestionAnswer(dspy.Signature):
    """Answer questions with short factual answers."""
    question = dspy.InputField()
    answer = dspy.OutputField()  #dxpy.OutputField(desc="often between 1 and 5 words")

class BasicQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(QuestionAnswer)
    
    def forward(self, question):
        return self.generate_answer(question=question)

# Trining.
trainset = [
    dspy.Example(question="Who is Tomboo-sama?", answer="Tomboo-sama lives in Tokyo.").with_inputs("question"),
    # ... more examples
]

# Define a metric
def validate_answer(example, pred, trace=None):
    return example.answer.lower() in pred.answer.lower()

# Compile/optimize the program
teleprompter = BootstrapFewShot(metric=validate_answer)
optimized_rag = teleprompter.compile(BasicQA(), trainset=trainset)

response = optimized_rag("Where does Tomboo-sama live?")

print("************:", response)
#************: Prediction(
#    reasoning='Based on the previous example provided, it was stated that "Tomboo-sama lives in Tokyo." This directly answers the question about where Tomboo-sama lives.',
#    answer='Tokyo.'
#)
