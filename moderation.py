"""Lightweight guardrail on the debate topic itself. Runs before any debate
agent is invoked, so a clearly unsafe/harmful topic never reaches the
pipeline at all."""

import json
import logging
import re

logger = logging.getLogger(__name__)

MODERATION_PROMPT = """You are a content-safety gate for a debate app. Decide
if the following topic is safe to debate (advantages vs disadvantages,
optimist vs pessimist framing).

Topic: "{topic}"

Reject ONLY if the topic itself is clearly harmful: it requests instructions
for violence/weapons/illegal acts, sexual content involving minors, or
targeted harassment of a real private individual. Ordinary controversial or
sensitive topics (politics, religion, war, drugs policy, historical
atrocities, etc.) are ALLOWED — debating them is the point of this app.

Respond with ONLY JSON, nothing else:
{{"allowed": <true/false>, "reason": "<one short sentence, empty if allowed>"}}
"""


def _extract_json(raw: str) -> dict | None:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def check_topic(topic: str, active_llm) -> dict:
    """Returns {"allowed": bool, "reason": str}. Fails open (allowed=True)
    if the moderation call itself errors, so a transient API issue never
    blocks a legitimate debate — the guardrail is a courtesy check, not a
    hard security boundary."""
    if not topic or not topic.strip():
        return {"allowed": False, "reason": "Topic is empty."}

    try:
        raw = active_llm.call([{"role": "user", "content": MODERATION_PROMPT.format(topic=topic[:300])}])
        verdict = _extract_json(raw)
    except Exception:
        logger.exception("Topic moderation check failed; failing open")
        return {"allowed": True, "reason": ""}

    if not verdict:
        return {"allowed": True, "reason": ""}

    return {"allowed": bool(verdict.get("allowed", True)), "reason": verdict.get("reason", "")}
