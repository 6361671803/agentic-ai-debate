from crewai import Task, Crew
from agents import optimist, pessimist, analyst, judge

# 🔥 ADDED: logging
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

print("\n===== MAIN EXECUTION STARTED (VERBOSE MODE) =====\n")  # 🔥 ADDED

topic = ""

print(f"📌 Topic: {topic}")  # 🔥 ADDED

task1 = Task(
    description=f"Give advantages of: {topic}",
    expected_output="List of advantages",
    agent=optimist
)

print("🟢 Task1 (Optimist) Created")  # 🔥 ADDED

task2 = Task(
    description=f"Give disadvantages of: {topic}",
    expected_output="List of disadvantages",
    agent=pessimist
)

print("🔴 Task2 (Pessimist) Created")  # 🔥 ADDED

task3 = Task(
    description="Compare both advantages and disadvantages",
    expected_output="Comparison analysis",
    agent=analyst
)

print("🟡 Task3 (Analyst) Created")  # 🔥 ADDED

task4 = Task(
    description="Choose best option and explain advantages. Return decision and confidence.",
    expected_output="Final decision with confidence",
    agent=judge
)

print("🔵 Task4 (Judge) Created\n")  # 🔥 ADDED

crew = Crew(
    agents=[optimist, pessimist, analyst, judge],
    tasks=[task1, task2, task3, task4],
    verbose=True
)

print("🚀 Crew Execution Started...\n")  # 🔥 ADDED

result = crew.kickoff()

print("\n===== FINAL RESULT =====\n")  # 🔥 ADDED
print(result)

print("\n===== EXECUTION COMPLETED =====\n")  # 🔥 ADDED
