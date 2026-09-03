# AI Debate Arena

A multi-agent AI debate system built with [CrewAI](https://www.crewai.com/). Five agents — Optimist, Pessimist, Analyst, Fact-Checker, and Judge — argue a user-supplied topic across four rounds, grounded in live web evidence and scored by an LLM judge, with the full run exportable as a citable PDF report.



Not just a prompt-chaining demo: it implements retrieval-augmented generation, real tool-calling agents, an LLM-as-judge evaluation pipeline, a self-critique/reflection pattern, automatic multi-provider failover, and a content-safety guardrail.

## How a debate works

1. **Round 1** — Optimist and Pessimist each argue independently, grounded in live-searched evidence for their side.
2. **Round 2 (Rebuttal)** — each responds to the other's points.
3. **Round 3 (Fact-Check + Analysis)** — the Fact-Checker agent picks the most checkable claims from both sides and verifies them itself via its own web-search tool call (SUPPORTED / UNVERIFIED / CONTRADICTED); the Analyst independently compares both sides.
4. **Verdict** — the Judge reads the analysis *and* the fact-check report and delivers a final decision.

Every agent turn is scored by a separate LLM-as-judge call (relevance / grounding / coherence / overall) and passes through a self-critique step before being finalized. The full run — arguments, scores, and sources — can be exported as a PDF.

## Architecture

| File | Responsibility |
|---|---|
| [`agents.py`](agents.py) | Defines the 5 agents; swappable LLM backend; automatic quota-fallback |
| [`app.py`](app.py) | Streamlit UI, debate orchestration, PDF export |
| [`rag.py`](rag.py) | Live web search → chunk → embed (Chroma) → retrieve, for grounding |
| [`tools.py`](tools.py) | Real CrewAI tools agents can call: `web_search`, `calculator` |
| [`evaluation.py`](evaluation.py) | LLM-as-judge scoring, logged to `evaluations.jsonl` |
| [`reflection.py`](reflection.py) | Self-critique / revise pass ("self-refine" pattern) |
| [`moderation.py`](moderation.py) | Content-safety gate on the topic before any agent runs |

### Design notes

- **No static knowledge base.** The debate topic is arbitrary and user-supplied, and often about current events, so grounding is done via *live* search at request time (Tavily, falling back to DuckDuckGo) rather than a pre-built vector index that would go stale or fail to cover an unseen topic.
- **Swappable LLM backend.** `LLM_PROVIDER` in `.env` selects Ollama (local, free), OpenAI, Gemini, or OpenRouter — same agent code, different backend, so cost/latency/quality tradeoffs are a config change, not a rewrite.
- **Automatic quota fallback.** If the primary provider hits a rate limit or daily quota mid-debate, the app detects it, switches every agent to an OpenRouter free-tier model, and retries — the debate keeps running instead of failing.
- **Reflection is context-aware.** The self-critique step only checks grounding when it was actually given retrieved evidence to check against — otherwise it only checks repetition/coherence, so it can't strip real citations from an agent (like the Fact-Checker) that found its own evidence via tool calls.
- **Judge score doubles as the confidence graph.** The bar chart plots real `evaluate()` output, not placeholder numbers.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:

```bash
LLM_PROVIDER=ollama   # or: openai, gemini, openrouter
```

Fill in the API key for whichever provider you choose (see `.env.example` for all options — Ollama needs no key, just a local `ollama serve`). Optionally add `OPENROUTER_API_KEY` regardless of your primary provider — it's used automatically as the quota-exhaustion fallback. Optionally add `TAVILY_API_KEY` for more reliable search (falls back to DuckDuckGo if unset or if Tavily's quota is exhausted).

## Run

```bash
streamlit run app.py
```

Enter a topic, click **Start Debate**, and watch it move through all four rounds.

## Tech stack

CrewAI · Streamlit · ChromaDB · ReportLab (PDF) · Tavily / DuckDuckGo (search) · Ollama / OpenAI / Gemini / OpenRouter (LLM backends)
