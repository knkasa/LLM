# Agentic RAG Gemini approach. Compare this with agentic_rag1.py

import os
import json
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.agents import ChatAgent
from camel.toolkits import FunctionTool

# 1. Setup Environment & Model Setup
# Ensure your OPENAI_API_KEY environment variable is set before running
model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O_MINI,
    model_config_dict={"temperature": 0.2}
)

# 2. Simulate a Vector Database with a "Hidden Gap" dependency
MOCK_VECTOR_STORE = {
    "project orion budget": (
        "Project Orion's budget was finalized at $5,000,000 for FY2026. "
        "The formal spending authorization rules and sign-off criteria are "
        "strictly governed by Protocol-X."
    ),
    "protocol-x": (
        "Protocol-X dictates that all capital allocations above $2M require "
        "written authorization and a formal signature from Director Sarah Lin."
    )
}

def query_vector_db(search_query: str) -> str:
    """Queries the vector database to retrieve context chunks based on a text string.
    
    Args:
        search_query (str): The keyword or term to look up in the vector database.
    """
    cleaned_query = search_query.lower()
    results = []
    for key, content in MOCK_VECTOR_STORE.items():
        if key in cleaned_query or any(word in cleaned_query for word in key.split()):
            results.append(content)
            
    if not results:
        return "No explicit matching chunks found in the database."
    return "\n---\n".join(results)

# Wrap our custom function into a CAMEL compatible tool interface
vector_tool = FunctionTool(query_vector_db)


# 3. Define System Archetypes
SUPERVISOR_SYSTEM = (
    "You are the Supervisor Agent. Your job is to coordinate an information "
    "gathering pipeline to answer user questions accurately based ONLY on provided facts. "
    "You will orchestrate interactions between the Data Extractor and the Reviewer."
)

VECTOR_AGENT_SYSTEM = (
    "You are the Vector DB Agent. Your sole responsibility is to extract "
    "raw information from the vector store using your query tool. "
    "Do not extrapolate. Return exactly what the database gives you."
)

REVIEWER_SYSTEM = (
    "You are the Reviewer Agent. Critically assess if the collected facts "
    "completely answer the user's initial question without any logical gaps. "
    "Look out for referenced entities, missing names, or codes that are left unexplained. "
    "If information is missing, output: [NEEDS_MORE: <describe what specific piece to search next>]. "
    "If everything is perfectly transparent and fully answered, output: [COMPLETE]."
)


# 4. Instantiate the Agent Crew
supervisor_agent = ChatAgent(system_message=SUPERVISOR_SYSTEM, model=model)
vector_agent = ChatAgent(system_message=VECTOR_AGENT_SYSTEM, model=model, tools=[vector_tool])
reviewer_agent = ChatAgent(system_message=REVIEWER_SYSTEM, model=model)


# 5. Execute Multi-Agent RAG Orchestration Loop
def run_agentic_rag(user_prompt: str, max_iterations: int = 3):
    print(f"🚀 Initial User Request: '{user_prompt}'\n")
    
    collected_context = []
    current_search_instruction = user_prompt
    
    for iteration in range(1, max_iterations + 1):
        print(f"--- 🔄 Iteration Loop {iteration} ---")
        
        # Step A: Vector DB Agent fetches information based on instructions
        vector_prompt = f"Find information related to: {current_search_instruction}"
        vector_res = vector_agent.step(vector_prompt)
        new_raw_facts = vector_res.msgs[0].content
        
        collected_context.append(new_raw_facts)
        print(f"🎨 [Vector DB Agent] Extracted Context:\n{new_raw_facts}\n")
        
        # Step B: Reviewer evaluates accumulated findings
        compiled_facts = "\n\n".join(collected_context)
        review_prompt = (
            f"Original User Question: {user_prompt}\n\n"
            f"Current Accumulated Facts:\n{compiled_facts}\n\n"
            "Evaluate completeness. Output exactly [COMPLETE] or [NEEDS_MORE: <topic>]."
        )
        reviewer_res = reviewer_agent.step(review_prompt)
        review_decision = reviewer_res.msgs[0].content.strip()
        print(f"🧐 [Reviewer Agent] Assessment: {review_decision}\n")
        
        if "[COMPLETE]" in review_decision:
            print("✅ Perfect! No more unknown variables detected.")
            break
        elif "[NEEDS_MORE:" in review_decision:
            # Extract the next missing node/entity suggested by reviewer
            # e.g., "[NEEDS_MORE: Protocol-X details]" -> "Protocol-X details"
            current_search_instruction = review_decision.split("[NEEDS_MORE:")[1].replace("]", "").strip()
            print(f"🔍 System flagged an unknown dependency. Re-routing search to target: '{current_search_instruction}'\n")
            
    # Step C: Final answer synthesis by the Supervisor
    final_prompt = (
        f"Synthesize a clear, cohesive final response to the user's question: '{user_prompt}' "
        f"using the entire verified history of gathered facts:\n\n{compiled_facts}"
    )
    final_res = supervisor_agent.step(final_prompt)
    
    print("--- 🏁 Final Integrated Response ---")
    print(final_res.msgs[0].content)

# Execute the workflow
if __name__ == "__main__":
    # This query directly highlights the issue: budget info is found in chunk 1, 
    # but the approver is locked inside the unknown entity 'Protocol-X'
    target_query = "What is the budget for Project Orion and exactly who approved it?"
    run_agentic_rag(user_prompt=target_query)