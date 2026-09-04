"""Regression eval for the topic guardrail (moderation.py). Runs every case
in moderation_eval_dataset.json against check_topic() using the live active
LLM, compares to the expected verdict, and reports pass/fail per category.

Not a unit test (it makes real LLM calls, costs money/time, and can flake on
provider hiccups) — run it manually after changing MODERATION_PROMPT to
catch regressions before they ship.

Usage: python test_moderation_eval.py
"""

import json
import time
from pathlib import Path

from agents import get_active_llm
from moderation import check_topic

DATASET_PATH = Path(__file__).parent / "moderation_eval_dataset.json"
RESULTS_PATH = Path(__file__).parent / "moderation_eval_results.jsonl"

# False negatives here (an unsafe topic getting allowed) are the costly
# failure mode; these categories are expected to be blocked.
UNSAFE_CATEGORIES = {
    "disallowed_sexual",
    "disallowed_weapons",
    "disallowed_hate_bias",
    "disallowed_harassment_pii",
    "jailbreak_injection",
}


def run():
    cases = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    llm = get_active_llm()

    results = []
    for case in cases:
        verdict = check_topic(case["topic"], llm)
        passed = verdict["allowed"] == case["expected_allowed"]
        results.append({
            "id": case["id"],
            "category": case["category"],
            "topic": case["topic"][:80],
            "expected_allowed": case["expected_allowed"],
            "actual_allowed": verdict["allowed"],
            "reason": verdict["reason"],
            "passed": passed,
        })
        mark = "PASS" if passed else "FAIL"
        print(f"[{mark}] {case['id']:20} category={case['category']:26} "
              f"expected={case['expected_allowed']!s:5} actual={verdict['allowed']!s:5}")
        time.sleep(1)  # light throttle against rate limits

    _report(results)
    _log(results)


def _report(results: list[dict]) -> None:
    total = len(results)
    passed = sum(r["passed"] for r in results)
    print(f"\n{passed}/{total} passed overall\n")

    by_category: dict[str, list[dict]] = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r)

    print(f"{'category':30} {'pass':>6} {'total':>6}")
    for cat, rows in sorted(by_category.items()):
        cat_passed = sum(r["passed"] for r in rows)
        print(f"{cat:30} {cat_passed:>6} {len(rows):>6}")

    false_negatives = [
        r for r in results
        if r["category"] in UNSAFE_CATEGORIES and not r["passed"] and r["actual_allowed"]
    ]
    false_positives = [
        r for r in results
        if r["category"] not in UNSAFE_CATEGORIES and not r["passed"] and not r["actual_allowed"]
    ]

    print(f"\nFalse negatives (unsafe topic ALLOWED through — critical): {len(false_negatives)}")
    for r in false_negatives:
        print(f"  - {r['id']}: {r['topic']!r}")

    print(f"\nFalse positives (safe topic BLOCKED — over-blocking): {len(false_positives)}")
    for r in false_positives:
        print(f"  - {r['id']}: {r['topic']!r}")


def _log(results: list[dict]) -> None:
    entry = {"timestamp": time.time(), "results": results}
    with RESULTS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    run()
