from crewai import Agent, LLM

# LLM configuration (shared)
llm = LLM(
    model="groq/llama-3.1-8b-instant",
    api_key="YOUR_API_KEY",
    temperature=0.3
)

# Optimist Agent
optimist = Agent(
    role="Optimist",
    goal="Give advantages of the topic",
    backstory="A positive thinker who highlights benefits",
    llm=llm
)

# Pessimist Agent
pessimist = Agent(
    role="Pessimist",
    goal="Give disadvantages of the topic",
    backstory="A critical thinker who finds flaws",
    llm=llm
)

# Analyst Agent
analyst = Agent(
    role="Analyst",
    goal="Compare both sides and analyze logically",
    backstory="A neutral evaluator using logic",
    llm=llm
)

# Judge Agent
judge = Agent(
    role="Judge",
    goal="Make final decision",
    backstory="A decision maker who selects the winner",
    llm=llm
)
