import os
import sys
import logging

# Windows consoles default to cp1252, which can't encode the emoji CrewAI's
# own internal logging writes to stdout/stderr — this crashes those log
# handlers (harmlessly, but noisily) before any of our code even runs.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Keep debate content local: don't upload execution traces/telemetry to
# CrewAI's cloud (app.crewai.com) by default.
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")

from dotenv import load_dotenv
from crewai import Agent, LLM

from tools import web_search, calculator

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------- SWAPPABLE LLM BACKEND ----------
# Set LLM_PROVIDER in .env to "ollama" (default, local/free), "openai", "gemini", or "openrouter".
PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()


def build_llm(provider: str) -> LLM:
    if provider == "openai":
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in .env")
        return LLM(model=f"openai/{model}", api_key=api_key, temperature=0.3, max_tokens=500)

    if provider == "gemini":
        model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set in .env")
        return LLM(model=f"gemini/{model}", api_key=api_key, temperature=0.3, max_tokens=500)

    if provider == "openrouter":
        # A free-tier model (":free" suffix) so this backend costs nothing,
        # same as the Ollama default. Swap OPENROUTER_MODEL in .env for any
        # other model from https://openrouter.ai/models.
        model = os.getenv("OPENROUTER_MODEL", "minimax/minimax-m3:free")
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set in .env")
        return LLM(model=f"openrouter/{model}", api_key=api_key, temperature=0.3, max_tokens=500)

    # Default: local Ollama, no API key / cost
    return LLM(
        model=os.getenv("OLLAMA_MODEL", "ollama/qwen2.5:3b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0.3,
        max_tokens=500,
    )


llm = build_llm(PROVIDER)
logger.info("LLM backend ready: provider=%s model=%s", PROVIDER, getattr(llm, "model", "unknown"))

# ---------- QUOTA FALLBACK ----------
# If the primary provider hits its daily quota / rate limit mid-session,
# automatically fail over to OpenRouter's free tier instead of just
# showing an error, so the debate can keep running.
FALLBACK_PROVIDER = "openrouter"
fallback_llm = None
if PROVIDER != FALLBACK_PROVIDER and os.getenv("OPENROUTER_API_KEY"):
    try:
        fallback_llm = build_llm(FALLBACK_PROVIDER)
        logger.info("Fallback LLM ready: provider=%s model=%s", FALLBACK_PROVIDER, getattr(fallback_llm, "model", "unknown"))
    except RuntimeError:
        fallback_llm = None

_using_fallback = False

TOOLS = [web_search, calculator]

optimist = Agent(
    role="Optimist",
    goal="Find advantages",
    backstory="Positive thinker",
    llm=llm,
    tools=TOOLS,
    verbose=True,
)

pessimist = Agent(
    role="Pessimist",
    goal="Find risks",
    backstory="Negative thinker",
    llm=llm,
    tools=TOOLS,
    verbose=True,
)

analyst = Agent(
    role="Analyst",
    goal="Compare both",
    backstory="Logical evaluator",
    llm=llm,
    tools=TOOLS,
    verbose=True,
)

judge = Agent(
    role="Judge",
    goal="Make decision",
    backstory="Final decision maker",
    llm=llm,
    tools=TOOLS,
    verbose=True,
)

fact_checker = Agent(
    role="Fact-Checker",
    goal="Verify whether specific claims made in the debate are supported by real evidence",
    backstory="Skeptical, evidence-driven verifier who trusts sources over rhetoric",
    llm=llm,
    tools=TOOLS,
    verbose=True,
)

ALL_AGENTS = [optimist, pessimist, analyst, judge, fact_checker]


_QUOTA_SIGNALS = ("429", "resource_exhausted", "quota", "rate limit", "rate_limit")


def is_quota_error(exc: Exception) -> bool:
    return any(signal in str(exc).lower() for signal in _QUOTA_SIGNALS)


def switch_to_fallback() -> bool:
    """Point every agent at the OpenRouter fallback LLM. Returns False if
    there's no fallback configured or it's already active."""
    global _using_fallback
    if fallback_llm is None or _using_fallback:
        return False
    for agent in ALL_AGENTS:
        agent.llm = fallback_llm
    _using_fallback = True
    logger.warning(
        "Primary LLM (%s) hit a quota/rate limit — switched all agents to OpenRouter fallback (%s)",
        PROVIDER, os.getenv("OPENROUTER_MODEL", "minimax/minimax-m3:free"),
    )
    return True


def is_using_fallback() -> bool:
    return _using_fallback


def get_active_llm() -> LLM:
    """The LLM currently powering the agents — the fallback once switched,
    otherwise the configured primary. Use this for any one-off LLM call
    (e.g. evaluation) that should track the same failover as the agents."""
    return fallback_llm if _using_fallback else llm
