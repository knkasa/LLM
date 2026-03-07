# Semantic kernel example.
# Semantic kernel is just like LangChain, but it can easily invoke Azure service.

#========== simple example ==============================
import asyncio
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

kernel = Kernel()

kernel.add_service(
    OpenAIChatCompletion(
        service_id="chat",
        ai_model_id="gpt-4o-mini",
        api_key="OPENAI_KEY"
    )
)

prompt = "Tell me a fun fact about {{$topic}}."

func = kernel.create_function_from_prompt(prompt)

async def main():
    result = await kernel.invoke(func, topic="Tokyo")
    print(result)

asyncio.run(main())

#========== Memory ================================
# Note: you can store these in vector DB such as Azure service, or Chroma, ...
kernel.memory.save_information(
    collection="notes",
    id="1",
    text="Tokyo is the capital of Japan"
)

results = await kernel.memory.search(
    "notes",
    "capital of Japan"
)

#========= system prompt ==========================
prompt = """
Summarize the following text:

{{$input}}
"""
from semantic_kernel.prompt_template import PromptTemplate
func = kernel.create_function_from_prompt(prompt)

result = await kernel.invoke(func, input="long text...")

#========= Tools ===================================
from semantic_kernel.functions import kernel_function

class MathPlugin:
    @kernel_function
    def add(self, a:int, b:int) -> int:
        return a + b
        
kernel.add_plugin(MathPlugin(), "math")