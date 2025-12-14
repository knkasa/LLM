# Example of using LangGraph react agent.
# It will reply Age, weather, multiplication.
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

@tool
def calculate_age(year: int) -> str:
    current_year = 2024
    return f"Age: {current_year - year}"

@tool
def get_weather(city: str) -> str:
    weather = {
        "delhi": "Hot",
        "berlin": "Mild",
        "sydney": "Windy"
    }
    return weather.get(city.lower(), "No data")

@tool
def multiply(a: float, b: float) -> str:
    return str(a * b)

tools = [calculate_age, get_weather, multiply]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

agent = create_react_agent(
    model=llm,
    tools=tools,
    state_modifier="Respond concisely. Use tools when needed."
)

if __name__ == "__main__":
    print("Agent ready. Type 'quit' to exit.")
    history = []
    while True:
        msg = input("\nYou: ")
        if msg.lower() in ["quit", "exit"]:
            break
        history.append(("user", msg))
        resp = agent.invoke({"messages": history})
        out = resp["messages"][-1]
        print("Agent:", out.content)
        history.append(("assistant", out.content))
 
