"""LLM-as-judge evaluation of each agent turn, logged to disk so output
quality can be tracked across runs instead of just eyeballing demos."""

import json
import logging
import re
import time
from pathlib import Path

logger = logging.getLogger(__name__)

LOG_PATH = Path(__file__).parent / "evaluations.jsonl"

RUBRIC_PROMPT = """You are an impartial evaluator judging one participant's turn in a debate.

Topic: {topic}
Role: {role}
Response:
\"\"\"{text}\"\"\"

Score the response from 1-10 on each dimension:
- relevance: does it directly address the topic?
- grounding: does it cite concrete evidence/facts rather than vague claims?
- coherence: is it well-structured and free of repetition?

Respond with ONLY valid JSON, no other text, in exactly this shape:
{{"relevance": <int 1-10>, "grounding": <int 1-10>, "coherence": <int 1-10>, "overall": <int 1-10>, "rationale": "<one short sentence>"}}
"""

_DEFAULT_SCORE = {
    "relevance": 0,
    "grounding": 0,
    "coherence": 0,
    "overall": 0,
    "rationale": "evaluation unavailable",
}


def _extract_json(raw: str) -> dict | None:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def evaluate(role: str, topic: str, text: str, judge_llm) -> dict:
    """Score one agent's response with a rubric-based LLM-as-judge call.
    Never raises — on any failure, returns a zeroed score so the debate UI
    keeps working even if the judge call fails."""

    prompt = RUBRIC_PROMPT.format(topic=topic, role=role, text=text[:2000])

    try:
        raw = judge_llm.call([{"role": "user", "content": prompt}])
        scores = _extract_json(raw) or dict(_DEFAULT_SCORE)
    except Exception:
        logger.exception("Evaluation failed for role=%s", role)
        scores = dict(_DEFAULT_SCORE)

    for key in ("relevance", "grounding", "coherence", "overall"):
        scores[key] = int(scores.get(key, 0)) if str(scores.get(key, "")).strip().isdigit() else 0
    scores.setdefault("rationale", "")

    _log(role, topic, scores)
    return scores


def _log(role: str, topic: str, scores: dict) -> None:
    entry = {"timestamp": time.time(), "role": role, "topic": topic, **scores}
    try:
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        logger.exception("Could not write evaluation log")
