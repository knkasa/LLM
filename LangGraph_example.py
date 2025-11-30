# LangGraph example

from langgraph.graph import StateGraph, END
from langchain.chat_models import ChatOpenAI
from typing import TypedDict, List

llm = ChatOpenAI(model="gpt-4o")

# Define the ouput structure in dict.
class State(TypedDict):
    question: str
    steps: List[str]
    result: str
    done: bool

# ------------- Nodes ----------------
def planner(state:State):
    """Break question into reasoning steps."""
    prompt = f"Break the problem into the next reasoning step:\nQuestion: {state['question']}\nSteps so far: {state['steps']}"
    output = llm.invoke(prompt).content
    state["steps"].append(output)
    return state

def worker(state: State):
    """Work on the latest step."""
    last_step = state["steps"][-1]
    prompt = f"Execute this step: {last_step}"
    result = llm.invoke(prompt).content
    state["result"] = result
    return state

def checker(state: State):
    """Check if the reasoning is complete."""
    prompt = f"Is the problem solved with this result?\nResult: {state['result']}\nAnswer yes or no and explain."
    evaluation = llm.invoke(prompt).content.lower()
    
    if "yes" in evaluation:
        state["done"] = True
    else:
        state["done"] = False
    return state

#------------- Graph ----------------
workflow = StateGraph(State)

# Nodes
workflow.add_node("planner", planner)
workflow.add_node("worker", worker)
workflow.add_node("checker", checker)

# Starting node.
workflow.set_entry_point("planner")

# Connect between nodes
workflow.add_edge("planner", "worker")
workflow.add_edge("worker", "checker")

# Define the last node for exit sign.
workflow.add_conditional_edges(
    "checker",
    lambda s: "finish" if s["done"] else "continue",
    {"finish": END, "continue":"planner"}  # End is imported from module.
    )

app = workflow.compile()

#------------- Run ----------------
result = app.invoke({"question": "How can we reduce global plastic pollution?", "steps": [], "result": "", "done": False})
print(result)
