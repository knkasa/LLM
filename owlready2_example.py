# Create Ontology, save in persistent memory, query it by LLM.
# pip install langgraph langchain openai owlready2

#=========== Create Ontolog and save it in local. ===========
from owlready2 import *

default_world.set_backend(filename="kg.sqlite3")

onto = get_ontology("http://example.org/company.owl")

with onto:
    class Person(Thing): pass
    class Skill(Thing): pass

    class worksOn(ObjectProperty):
        domain = [Person]
        range = [Skill]

ken = Person("Ken")
ts = Skill("TimeSeriesForecasting")
ken.worksOn.append(ts)

onto.save()

#========= LangGraph setup =====================
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain.chat_models import ChatOpenAI
from owlready2 import default_world

# ---- LLM ----
llm = ChatOpenAI(model="gpt-4o-mini")

# ---- State ----
class GraphState(TypedDict):
    question: str
    sparql_query: str
    query_result: str
    final_answer: str

# ---- Node 1: Planner (Generate SPARQL) ----
def planner_node(state: GraphState):
    prompt = f"""
You are a SPARQL generator.
Ontology contains:
- Class: Person
- Class: Skill
- Property: worksOn

Generate a valid SPARQL query.
Question: {state['question']}
Only return SPARQL.
"""
    response = llm.invoke(prompt)
    return {"sparql_query": response.content}

# ---- Node 2: Executor ----
def executor_node(state: GraphState):
    query = state["sparql_query"]
    try:
        results = list(default_world.sparql(query))
        return {"query_result": str(results)}
    except Exception as e:
        return {"query_result": f"ERROR: {e}"}

# ---- Node 3: Answer Generator ----
def answer_node(state: GraphState):
    prompt = f"""
Question: {state['question']}

SPARQL Result:
{state['query_result']}

Generate a natural language answer.
"""
    response = llm.invoke(prompt)
    return {"final_answer": response.content}

# ---- Build Graph ----
workflow = StateGraph(GraphState)

workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("answer", answer_node)

workflow.set_entry_point("planner")

workflow.add_edge("planner", "executor")
workflow.add_edge("executor", "answer")
workflow.add_edge("answer", END)

app = workflow.compile()

#---- Run it -------
result = app.invoke({
    "question": "Who works on TimeSeriesForecasting?"
})

print(result["final_answer"])

