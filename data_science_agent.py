# https://koshurai.medium.com/your-data-science-workflow-is-already-obsolete-heres-how-agentic-ai-changes-everything-in-2026-d2d50888a66d
# Data science agent.

# Agentic experiment manager (LangChain + custom tools)
from langchain.agents import initialize_agent, Tool
from langchain_openai import ChatOpenAI

def read_metrics(run_id: str) -> dict:
    """Fetch current val_loss, accuracy from experiment tracker."""
    return experiment_tracker.get_metrics(run_id)
def adjust_lr(run_id: str, new_lr: float) -> str:
    """Modify learning rate mid-training via API."""
    return experiment_tracker.update_config(run_id, {"lr": new_lr})
def stop_run(run_id: str, reason: str) -> str:
    """Halt a training run and log the reason."""
    return experiment_tracker.stop(run_id, reason=reason)
def notify_human(message: str) -> str:
    """Send Slack alert when agent needs human judgment."""
    return slack_client.post(channel="#ml-ops", text=message)
tools = [
    Tool(name="read_metrics", func=read_metrics,
         description="Read current training metrics for a given run"),
    Tool(name="adjust_lr",    func=adjust_lr,
         description="Adjust learning rate if loss is plateauing"),
    Tool(name="stop_run",     func=stop_run,
         description="Stop a run if loss is diverging or NaN"),
    Tool(name="notify_human", func=notify_human,
         description="Alert the team for decisions beyond agent scope"),
]
agent = initialize_agent(
    tools=tools,
    llm=ChatOpenAI(model="gpt-4o"),
    agent="zero-shot-react-description",
    verbose=True
)
# Agent now runs every 15 minutes via cron
agent.run(
    f"Monitor run {run_id}. If val_loss increases 3 consecutive "
    f"epochs, halve the LR. If loss is NaN, stop and notify the team. "
    f"If accuracy > 0.92, notify the team it is ready for review."
)


from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
# Define state schema — every node reads and writes this
class PipelineState(TypedDict):
    raw_data: str
    cleaned_data: str
    model_metrics: dict
    analysis_report: str
    needs_human_review: bool
def ingest_data(state: PipelineState) -> PipelineState:
    """Node 1: Pull latest data from warehouse."""
    raw = data_warehouse.fetch(query="SELECT * FROM events WHERE dt = today()")
    return {"raw_data": raw}
def clean_and_validate(state: PipelineState) -> PipelineState:
    """Node 2: Run data quality checks, flag issues."""
    cleaned, issues = data_quality_agent.run(state["raw_data"])
    return {
        "cleaned_data": cleaned,
        "needs_human_review": len(issues) > 0
    }
def run_model(state: PipelineState) -> PipelineState:
    """Node 3: Score cleaned data through production model."""
    metrics = model.predict_and_evaluate(state["cleaned_data"])
    return {"model_metrics": metrics}
def generate_report(state: PipelineState) -> PipelineState:
    """Node 4: LLM writes the executive summary."""
    report = llm.invoke(
        f"Write a concise data quality and model performance report "
        f"based on these metrics: {state['model_metrics']}"
    )
    return {"analysis_report": report.content}
# Routing logic: branch based on data quality flag
def route_after_cleaning(state: PipelineState) -> str:
    if state["needs_human_review"]:
        return "human_review"   # Pause and wait for approval
    return "run_model"          # Continue automatically
# Build the graph
workflow = StateGraph(PipelineState)
workflow.add_node("ingest",          ingest_data)
workflow.add_node("clean",           clean_and_validate)
workflow.add_node("run_model",       run_model)
workflow.add_node("generate_report", generate_report)
workflow.add_node("human_review",    human_review_node)
workflow.set_entry_point("ingest")
workflow.add_edge("ingest", "clean")
workflow.add_conditional_edges("clean", route_after_cleaning)
workflow.add_edge("run_model",       "generate_report")
workflow.add_edge("generate_report", END)
# Compile with checkpointing — agent resumes after crashes
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string("agent_state.db")
app = workflow.compile(checkpointer=checkpointer)
# Run it
result = app.invoke({"raw_data": "", "needs_human_review": False})
print(result["analysis_report"])



from crewai import Agent, Task, Crew, Process

# Define specialist agents with roles and tools
data_engineer = Agent(
    role="Senior Data Engineer",
    goal="Ensure data pipelines are clean, complete, and well-documented",
    backstory="10 years building robust ETL systems for fintech companies.",
    tools=[sql_tool, data_quality_tool],
    verbose=True
)
ml_engineer = Agent(
    role="ML Engineer",
    goal="Train, evaluate, and recommend the best model for the use case",
    backstory="Specialises in production ML with a focus on explainability.",
    tools=[training_tool, evaluation_tool, shap_tool],
    verbose=True
)
analyst = Agent(
    role="Senior Data Analyst",
    goal="Translate model outputs into clear business recommendations",
    backstory="Expert at turning statistical results into executive-ready insights.",
    tools=[visualisation_tool, report_tool],
    verbose=True
)
# Define tasks (sequentially assigned to agents)
prepare_data_task = Task(
    description="Pull churn data for Q1 2026. Clean nulls, encode categoricals, "
                "run quality checks. Return a summary of any data issues found.",
    agent=data_engineer,
    expected_output="Clean dataset + data quality report"
)
train_model_task = Task(
    description="Train a churn prediction model on the prepared dataset. "
                "Try XGBoost and LightGBM. Compare AUC-ROC and F1. "
                "Generate SHAP values for the top 10 features.",
    agent=ml_engineer,
    expected_output="Model comparison table + SHAP feature importance chart"
)
report_task = Task(
    description="Write a 1-page executive summary of the churn model results. "
                "Include top drivers of churn, model performance, and 3 recommended "
                "actions for the retention team.",
    agent=analyst,
    expected_output="Executive summary (markdown format, <400 words)"
)
# Assemble and run the crew
crew = Crew(
    agents=[data_engineer, ml_engineer, analyst],
    tasks=[prepare_data_task, train_model_task, report_task],
    process=Process.sequential,
    verbose=True
)
result = crew.kickoff()
print(result.raw)


