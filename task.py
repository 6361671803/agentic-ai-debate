from crewai import Task
from agents import optimist, pessimist, analyst, judge

optimist_task = Task(
    description="""
Choose the best option and explain advantages.
Return decision and confidence.
""",
    expected_output="Decision with confidence score and reasoning",
    agent=optimist
)

pessimist_task = Task(
    description="""
Find risks and choose best option.
Return decision and confidence.
""",
    expected_output="Decision with confidence score and reasoning",
    agent=pessimist
)

analyst_task = Task(
    description="""
Compare both options logically.
Return decision and confidence.
""",
    expected_output="Logical comparison with confidence score",
    agent=analyst
)

judge_task = Task(
    description="""
Read all agent outputs and choose final decision.
""",
    expected_output="Final decision with explanation",
    agent=judge
)
