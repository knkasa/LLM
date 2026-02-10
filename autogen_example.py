import autogen

config_list = [
    {
        "model": "gpt-4",
        "api_key": "YOUR_OPENAI_API_KEY"
    }
]

#Define helper agents.
assistant = autogen.AssistantAgent(
    name="assistant",  # Name can be anything.
    system_message="You are a harsh code revier.",
    llm_config={"config_list": config_list}
)

#Create the UserProxy (The "Executor")
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",  # Set to "ALWAYS" if you want to approve every line
    max_consecutive_auto_reply=10,
    code_execution_config={
        "work_dir": "coding_scripts", # Files will be saved here
        "use_docker": False           # Runs locally on your machine
    }
)

#Start the conversation
user_proxy.initiate_chat(
    [assistant],  # Define helper agents 
    message="Find out what day of the week it will be on June 15, 2030 and print it."
)

#Tools
# A simple tool
def get_weather(location):
    return f"The weather in {location} is 25°C and sunny."

# Registering the tool (Simplified syntax)
autogen.agentchat.register_function(
    get_weather,
    caller=assistant,  # The one who decides to use it
    executor=user_proxy, # The one who actually runs it
    name="get_weather",
    description="Returns the weather for a given city"
)

#RAG
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

rag_agent = RetrieveUserProxyAgent(
    name="Retriever",
    retrieve_config={
        "task": "code",
        "docs_path": "./my_docs", # Folder containing your PDFs/Docs
        "chunk_token_size": 2000,
        "vector_db": "chroma",     # Automatically sets up a local DB
    }
)


# 4. Start the conversation
user_proxy.initiate_chat(
    assistant, 
    message="Find out what day of the week it will be on June 15, 2030 and print it."
)

