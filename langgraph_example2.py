# https://kadermiyanyedi.medium.com/building-an-ocr-react-agent-from-scratch-with-langgraph-and-gemini-8884b4fbde00?source=email-00a0275a2be7-1773165050524-digest.reader--8884b4fbde00----8-98------------------688df3cd_e0fc_4e67_86a7_f362d39e45ec-1

from typing import TypedDict
from langgraph.graph import StateGraph, END, START

class AgentState(TypedDict):
    number: int
    result: int | None

def get_number(state: AgentState):
    state["number"] = int(input("Enter a number: "))
    return state

def square_number(state: AgentState):
    state["result"] = state["number"] ** 2
    print(f"Even number detected. Result: {state['result']}")
    return state

def cube_number(state: AgentState):
    state["result"] = state["number"] ** 3
    print(f"Odd number detected. Result: {state['result']}")
    return state

def should_continue(state: AgentState):
    if state["number"] % 2 == 0:
        return "even"
    return "odd"

graph = StateGraph(AgentState)

graph.add_node("get_number", get_number)
graph.add_node("square", square_number)
graph.add_node("cube", cube_number)

graph.add_conditional_edges(
    "get_number",
    should_continue,
    {
        "even": "square",
        "odd": "cube",
    },
)

graph.add_edge(START, "get_number")
graph.add_edge("square", END)
graph.add_edge("cube", END)

app = graph.compile()
app.invoke({"number": 0, "result": None})
