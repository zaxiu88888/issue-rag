"""LLM-as-judge scoring with an explicit rubric.

Judge design notes
------------------
* The judge sees the **reference answer**, so it scores correctness against
  something concrete instead of vibes.
* Answers are graded on a three point scale rather than binary: partial credit
  distinguishes "vague but not wrong" from "confidently wrong", and confident
  wrongness is the failure mode this project measures.
* Output is forced into JSON and parsed defensively; a local 3B model will
  occasionally wrap it in prose.
* A human spot check of ~30 items is still required before publishing numbers.
  LLM judges have known biases and a README should say so.

The grounded variant additionally receives the retrieved context and reports
whether every claim in the answer is supported by it -- that is the citation
fidelity / hallucination signal.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .generate import ChatModel

VERDICTS = ("correct", "partial", "incorrect")

GRADE_SYSTEM = (
    "You are grading a technical support answer. Be strict and literal.\n"
    "You will see a QUESTION, a REFERENCE answer written by the project "
    "maintainers, and a CANDIDATE answer.\n\n"
    "Grade the CANDIDATE on correctness against the REFERENCE:\n"
    '- "correct": the candidate states the same resolution or fact as the '
    "reference. Wording may differ. Extra harmless detail is fine.\n"
    '- "partial": the candidate is directionally right but misses the key step, '
    "names the wrong API, or is too vague to act on.\n"
    '- "incorrect": the candidate contradicts the reference, invents a fix, or '
    "invents a version number.\n\n"
    "A confident but wrong version number, file name, or API name is "
    '"incorrect", never "partial".\n\n'
    "Reply with JSON only, no prose:\n"
    '{"verdict": "correct|partial|incorrect", "fabricated": true|false, '
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


class Judge:
    def __init__(self, model: ChatModel, *, max_new_tokens: int = 160) -> None:
        self.model = model
        self.max_new_tokens = max_new_tokens

    def grade(
        self, question: str, reference: str, candidate: str
    ) -> Grade:
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
        verdict = str(parsed.get("verdict", "")).strip().lower()
        if verdict not in VERDICTS:
            verdict = "unparsed"
        return Grade(
            verdict=verdict,
            fabricated=bool(parsed.get("fabricated", False)),
            reason=str(parsed.get("reason", ""))[:400],
            raw=result.text,
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
