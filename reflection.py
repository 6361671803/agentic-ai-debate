"""Self-critique / reflection ("self-refine"): before an agent's answer is
finalized, it critiques its own draft against the retrieved evidence and
revises once if the critique finds a genuine issue."""

import json
import logging
import re

logger = logging.getLogger(__name__)

CRITIQUE_PROMPT = """You are critiquing your own draft argument before finalizing it.

Topic: {topic}

Retrieved evidence available to you:
{context}

Your draft:
\"\"\"{draft}\"\"\"

Check for genuine problems only:
- repetition (the same point restated more than once)
- incoherence (contradicts itself or is poorly structured){grounding_check}

Respond with ONLY JSON, nothing else:
{{"has_issues": <true/false>, "issues": ["<short issue>", ...]}}
"""

_GROUNDING_CHECK = "\n- ungrounded claims (asserted as fact but not supported by the evidence above)"

REVISE_PROMPT = """Revise your own draft to fix these specific issues: {issues}

Topic: {topic}

Retrieved evidence:
{context}

Original draft:
\"\"\"{draft}\"\"\"

Fix ONLY the listed issues — do not remove specific figures, statistics, or
citations that aren't part of the listed issues, even if no evidence was
retrieved for this check; this check has no way to confirm whether your
draft's own sources (e.g. from a tool call you made) are valid, so absence
of evidence here is not grounds to strip them.

Output ONLY the revised argument, keeping the same format/structure as the
original (numbered points), nothing else — no preamble, no explanation of
what you changed.
"""


def _extract_json(raw: str) -> dict | None:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def critique_and_revise(role: str, topic: str, draft: str, context: str, active_llm) -> dict:
    """Returns {"text": str, "revised": bool, "issues": list[str]}.
    Never raises — on any failure, returns the original draft unchanged."""
    if not draft or not draft.strip():
        return {"text": draft, "revised": False, "issues": []}

    has_context = bool(context and context.strip())

    try:
        critique_raw = active_llm.call([{
            "role": "user",
            "content": CRITIQUE_PROMPT.format(
                topic=topic,
                context=context or "(no evidence was pre-fetched for this check)",
                draft=draft[:2000],
                grounding_check=_GROUNDING_CHECK if has_context else "",
            ),
        }])
        critique = _extract_json(critique_raw)
    except Exception:
        logger.exception("Self-critique failed for role=%s", role)
        return {"text": draft, "revised": False, "issues": []}

    if not critique or not critique.get("has_issues"):
        return {"text": draft, "revised": False, "issues": []}

    issues = critique.get("issues", [])

    try:
        revised_raw = active_llm.call([{
            "role": "user",
            "content": REVISE_PROMPT.format(
                issues="; ".join(issues) or "general quality",
                topic=topic,
                context=context or "(no evidence was pre-fetched for this check)",
                draft=draft[:2000],
            ),
        }])
        if revised_raw and revised_raw.strip():
            return {"text": revised_raw.strip(), "revised": True, "issues": issues}
    except Exception:
        logger.exception("Self-revision failed for role=%s", role)

    return {"text": draft, "revised": False, "issues": issues}
