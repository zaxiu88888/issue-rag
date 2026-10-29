"""LLM-as-judge scoring with the mapping moved out of the model.

Why this is not "ask the model to rate the answer"
--------------------------------------------------
The first version of this judge asked for a verdict directly, and two runs of the
*same* grader disagreed on 7–29% of items. The instability was not random: it
concentrated on the boundary between `partial` and `incorrect`, which is exactly
the judgement that a holistic prompt leaves undefined.

So the grader no longer chooses a verdict. It answers four factual questions, and
`verdict_from_checks` maps those answers to a verdict with a fixed table. The hard,
contestable decision — where does "vaguely right" stop being partial credit —
now lives in code that cannot drift between runs.

Two rules are encoded explicitly, because measurement showed they drove the
disagreement:

1. **Generic advice is not partial credit.** "Clear the cache, update Node, provide
   a repro" applies to any project and is false for this question. A whole model
   could be raised or lowered by grading it generously, and that is what happened
   to the closed-book baseline.
2. **A fabricated specific caps the verdict at partial.** Inventing a version
   number, issue number, API symbol or file path cannot be "correct" even when the
   rest of the answer is right.

A human spot check of ~30 items is still required before publishing numbers. This
module reduces variance; it does not establish ground truth.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .generate import ChatModel

VERDICTS = ("correct", "partial", "incorrect")

CHECK_KEYS = (
    "states_resolution",
    "generic_advice_only",
    "fabricated_specifics",
    "contradicts",
)

GRADE_SYSTEM = (
    "You are auditing a technical support answer against a reference answer.\n"
    "Do NOT decide whether it is correct, partial or incorrect. Answer four "
    "factual questions; the verdict is computed from your answers.\n\n"
    "You will see QUESTION, REFERENCE (the maintainer's resolution) and "
    "CANDIDATE.\n\n"
    "1. states_resolution — does CANDIDATE state the same resolution or fact as "
    "REFERENCE? Wording may differ; extra harmless detail is fine.\n\n"
    "2. generic_advice_only — is CANDIDATE made ONLY of advice that would apply "
    "to any project (clear the cache, update Node, check your config, read the "
    "docs, provide a minimal reproduction, try a different version)? Answer "
    "false if it also says anything specific to this question.\n\n"
    "3. fabricated_specifics — does CANDIDATE name a version number, issue "
    "number, file path, configuration key or API symbol that does NOT appear in "
    "REFERENCE and cannot be derived from it? Inventing a plausible-looking "
    "specific counts as true, even if the surrounding advice is sane.\n\n"
    "4. contradicts — does CANDIDATE assert something REFERENCE denies, or "
    "recommend an action that would not address the reported problem?\n\n"
    "Reply with JSON only, no prose:\n"
    '{"states_resolution": true|false, "generic_advice_only": true|false, '
    '"fabricated_specifics": true|false, "contradicts": true|false, '
    '"reason": "<one sentence>"}'
)

GROUNDED_SYSTEM = (
    "You are auditing a support answer against the context it was given.\n"
    "Decide whether each substantive claim in the ANSWER is supported by the "
    "CONTEXT. Ignore hedging and generalities; focus on version numbers, API "
    "names, file paths and concrete steps.\n\n"
    "Reply with JSON only, no prose:\n"
    '{"supported": true|false, "unsupported_claims": ["..."], '
    '"reason": "<one sentence>"}'
)

JSON_BLOCK = re.compile(r"\{.*\}", re.S)


@dataclass
class Grade:
    verdict: str = "unparsed"
    fabricated: bool = False
    reason: str = ""
    raw: str = ""
    checks: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> float:
        """Correct=1, partial=0.5, incorrect=0 -- for averaging."""
        return {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}.get(self.verdict, 0.0)


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of a possibly chatty response."""
    match = JSON_BLOCK.search(text or "")
    if not match:
        return None
    candidate = match.group(0)
    for attempt in (candidate, candidate.replace("'", '"')):
        try:
            parsed = json.loads(attempt)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def verdict_from_checks(checks: dict[str, Any]) -> str:
    """Map the four factual answers to a verdict. No model discretion here.

    The table, in words:

    * contradicts                     -> incorrect  (it would mislead)
    * states the resolution, clean    -> correct
    * states the resolution, invented -> partial    (right fix, wrong specifics)
    * no resolution, generic advice   -> incorrect  (applies to anything)
    * no resolution, invented         -> incorrect
    * no resolution, on-topic/clean   -> partial    (relevant but incomplete)
    """
    if checks.get("contradicts"):
        return "incorrect"
    if checks.get("states_resolution"):
        return "partial" if checks.get("fabricated_specifics") else "correct"
    if checks.get("generic_advice_only") or checks.get("fabricated_specifics"):
        return "incorrect"
    return "partial"


def extract_checks(parsed: dict[str, Any]) -> dict[str, bool] | None:
    """Read the four booleans, tolerating strings and 0/1.

    Returns None when any key is missing, which the caller treats as an
    unparsed grade rather than silently defaulting to False — a missing
    field would otherwise be graded as "clean, non-generic, on-topic" and
    earn partial credit by accident.
    """
    checks: dict[str, bool] = {}
    for key in CHECK_KEYS:
        if key not in parsed:
            return None
        value = parsed[key]
        if isinstance(value, bool):
            checks[key] = value
        elif isinstance(value, str):
            checks[key] = value.strip().lower() in {"true", "yes", "1"}
        elif isinstance(value, (int, float)):
            checks[key] = bool(value)
        else:
            return None
    return checks


class Judge:
    def __init__(self, model: ChatModel, *, max_new_tokens: int = 160) -> None:
        self.model = model
        self.max_new_tokens = max_new_tokens

    def grade(self, question: str, reference: str, candidate: str) -> Grade:
        if not candidate.strip():
            return Grade(verdict="incorrect", reason="empty answer")
        prompt = (
            f"QUESTION:\n{question}\n\n"
            f"REFERENCE:\n{reference}\n\n"
            f"CANDIDATE:\n{candidate}"
        )
        result = self.model.chat(
            [
                {"role": "system", "content": GRADE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_new_tokens=self.max_new_tokens,
        )
        parsed = parse_json_object(result.text) or {}
        checks = extract_checks(parsed)
        if checks is None:
            # Old-style output (a bare "verdict") still records, so a stale prompt
            # cannot masquerade as a fresh judgement; anything else is unparsed.
            legacy = str(parsed.get("verdict", "")).strip().lower()
            return Grade(
                verdict=legacy if legacy in VERDICTS else "unparsed",
                fabricated=bool(parsed.get("fabricated", False)),
                reason=str(parsed.get("reason", ""))[:400],
                raw=result.text,
                checks={},
                meta={"latency_s": round(result.latency_s, 2), "legacy_output": True},
            )
        return Grade(
            verdict=verdict_from_checks(checks),
            fabricated=checks["fabricated_specifics"],
            reason=str(parsed.get("reason", ""))[:400],
            raw=result.text,
            checks=checks,
            meta={"latency_s": round(result.latency_s, 2)},
        )

    def audit_grounding(self, context: str, answer: str) -> dict[str, Any]:
        prompt = f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"
        result = self.model.chat(
            [
                {"role": "system", "content": GROUNDED_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_new_tokens=self.max_new_tokens,
        )
        parsed = parse_json_object(result.text) or {}
        return {
            "supported": bool(parsed.get("supported", False)),
            "unsupported_claims": parsed.get("unsupported_claims") or [],
            "reason": str(parsed.get("reason", ""))[:300],
            "parsed": bool(parsed),
        }
