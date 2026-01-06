# Create multi-agents.

from crewai import Agent, Task, Crew, Process

# Define agents
researcher = Agent(role="Researcher", goal="Gather facts about AI")

# Define tasks
facts_task = Task(description="Collect top 3 AI trends")

# Build a crew & run
my_crew = Crew(agents=[researcher], tasks=[facts_task], process=Process.sequential)
result = my_crew.kickoff()
print(result)
