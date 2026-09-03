"""CrewAI tools the agents can call directly (real function calling, not just
prompt-stuffed context)."""

import ast
import logging
import operator

from crewai.tools import tool

from rag import _live_search

logger = logging.getLogger(__name__)


@tool("web_search")
def web_search(query: str) -> str:
    """Search the live web for up-to-date facts, news, or evidence about a
    query. Use this to ground an argument in real, current information
    instead of guessing. Returns titles, short summaries, and source URLs."""
    results = _live_search(query, max_results=5)
    if not results:
        return "No search results found."
    lines = []
    for r in results:
        lines.append(f"- {r.get('title')}: {r.get('body')} (source: {r.get('href')})")
    return "\n".join(lines)


_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression")


@tool("calculator")
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression (+, -, *, /, ** and
    parentheses only). Use this for any numeric claim (percentages,
    comparisons, growth rates) instead of guessing the number."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
    except Exception:
        logger.exception("calculator tool failed for expression=%r", expression)
        return "Could not evaluate that expression."
