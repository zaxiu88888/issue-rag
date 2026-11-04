"""从维护者回复里挖问答对、经人工复核建评测集。

Mine question/answer pairs from harvested issues and build the eval set.

This is the stage that decides whether the project is worth anything, so the
filtering rules are explicit rather than clever.

The self-retrieval trap
-----------------------
If the query is the issue body, the retriever can match that very issue's own
documents and every metric looks wonderful while measuring nothing. So:

* the query is the issue **title** (a user-style problem statement), not the body;
* retrieval is reported both *with* the source issue in the index (the honest
  product number -- a real user should be shown the existing thread) and
  *leave-one-issue-out* (the strict number that shows generalisation).

Both numbers go in the results table. Reporting only the flattering one is the
kind of thing an interviewer will find.

Output
------
``evalset_candidates.jsonl``  machine-readable candidates
``evalset_review.csv``        one row per candidate for human keep/drop
``evalset.jsonl``             final set, produced by ``--finalize``
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .config import RepoPreset, load_preset
from .corpus import MAINTAINER_ROLES, clean_text, is_bot, load_jsonl
from .fetch import COMMENTS_FILE, ISSUES_FILE

CANDIDATES_FILE = "evalset_candidates.jsonl"
REVIEW_FILE = "evalset_review.csv"
EVALSET_FILE = "evalset.jsonl"

# Answers that carry no information: triage replies and bot noise.
LOW_VALUE_ANSWER = re.compile(
    r"^\s*(duplicate of|closing (this )?as|this is a duplicate|"
    r"please provide a (minimal )?reprod|"
    r"thanks for (the )?report|we'?ll take a look|fixed in v?\d+\.\d+\.\d+\s*$)",
    re.I,
)

VERSION_PATTERN = re.compile(r"\bv?\d+\.\d+\.\d+(?:-[\w.]+)?\b")
PATH_PATTERN = re.compile(r"\b[\w./-]+\.(?:ts|tsx|js|jsx|py|go|md)\b")
SYMBOL_PATTERN = re.compile(r"`[A-Za-z_][\w.]{2,}`")
CODE_BLOCK = re.compile(r"```")

# Thresholds calibrated per repository, not guessed. Vite maintainers reply
# tersely -- "This will be fixed by #23603", "CSS works with UTF-8 so I don't
# think we need to clarify this" -- and a 150-character floor rejected every
# real answer in the sample. Short is not the same as low-information; the
# triage-boilerplate filter below is what removes the useless ones.
MIN_ANSWER_CHARS = 60
MAX_ANSWER_CHARS = 6_000
MIN_TITLE_CHARS = 12


def answer_features(text: str) -> dict[str, Any]:
    """Cheap structural tags, used for stratification rather than scoring."""
    return {
        "has_version": bool(VERSION_PATTERN.search(text)),
        "has_file_path": bool(PATH_PATTERN.search(text)),
        "has_code_block": bool(CODE_BLOCK.search(text)),
        "has_symbol": bool(SYMBOL_PATTERN.search(text)),
        "chars": len(text),
    }


def normalise_title(title: str) -> str:
    return re.sub(r"\s+", " ", title.lower()).strip()


def collect_candidates(
    issues: list[dict[str, Any]],
    comments: list[dict[str, Any]],
    *,
    max_per_issue: int = 1,
    min_answer_chars: int = MIN_ANSWER_CHARS,
    stats: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Join issues to their best maintainer answer and keep plausible pairs.

    ``stats`` is filled with the per-stage counts so the report can show the
    funnel. The funnel is the honest answer to "why only N items?": every stage
    that discards data is numbered, rather than hidden behind a final count.
    """
    funnel = stats if stats is not None else {}
    funnel.update(
        {
            "comments_total": len(comments),
            "comments_human": sum(1 for c in comments if not is_bot(c.get("author"))),
            "comments_maintainer": sum(
                1
                for c in comments
                if c.get("author_association") in MAINTAINER_ROLES
                and not is_bot(c.get("author"))
            ),
        }
    )

    by_issue: dict[int, list[dict[str, Any]]] = {}
    for comment in comments:
        by_issue.setdefault(comment.get("issue_number"), []).append(comment)

    seen_titles: set[str] = set()
    candidates: list[dict[str, Any]] = []
    issues_in_scope = 0
    issues_with_maintainer = 0
    issues_with_usable_answer = 0
    dropped_triage = 0
    dropped_short = 0

    for issue in issues:
        if issue.get("is_pull_request"):
            continue
        title = clean_text(issue.get("title") or "")
        if len(title) < MIN_TITLE_CHARS:
            continue
        if "duplicate" in [lbl.lower() for lbl in (issue.get("labels") or [])]:
            continue
        issues_in_scope += 1
        key = normalise_title(title)
        if key in seen_titles:
            continue

        pool = [
            c
            for c in by_issue.get(issue["number"], [])
            if c.get("author_association") in MAINTAINER_ROLES
            and not is_bot(c.get("author"))
            and c.get("body")
        ]
        if not pool:
            continue
        issues_with_maintainer += 1

        # Prefer the longest maintainer answer that is not triage boilerplate.
        pool.sort(key=lambda c: (-(c.get("reactions") or 0), -len(c.get("body") or "")))
        chosen: list[dict[str, Any]] = []
        for comment in pool:
            body = clean_text(comment["body"])
            if len(body) < min_answer_chars:
                dropped_short += 1
                continue
            if LOW_VALUE_ANSWER.search(body) and len(body) < 400:
                dropped_triage += 1
                continue
            chosen.append({"comment": comment, "body": body})
            if len(chosen) >= max_per_issue:
                break
        if not chosen:
            continue
        issues_with_usable_answer += 1

        reference = "\n\n".join(item["body"] for item in chosen)
        if len(reference) > MAX_ANSWER_CHARS:
            reference = reference[:MAX_ANSWER_CHARS] + "\n[... truncated ...]"

        seen_titles.add(key)
        candidates.append(
            {
                "id": f"q{issue['number']}",
                "issue_number": issue["number"],
                "query": title,
                "query_mode": "title",
                "reference_answer": reference,
                "reference_comment_ids": [item["comment"]["id"] for item in chosen],
                "reference_authors": [item["comment"].get("author") for item in chosen],
                "issue_created_at": issue.get("created_at"),
                "issue_url": issue.get("html_url"),
                "labels": issue.get("labels") or [],
                "features": answer_features(reference),
                "keep": "",  # filled in by the human reviewer
                "review_notes": "",
            }
        )

    candidates.sort(key=lambda c: c.get("issue_created_at") or "", reverse=True)
    funnel.update(
        {
            "issues_in_scope": issues_in_scope,
            "issues_with_maintainer": issues_with_maintainer,
            "issues_with_usable_answer": issues_with_usable_answer,
            "comments_rejected_short": dropped_short,
            "comments_rejected_triage": dropped_triage,
            "candidates": len(candidates),
        }
    )
    return candidates


def write_review_sheet(candidates: list[dict[str, Any]], path: Path) -> None:
    """A CSV a human can actually work through in twenty minutes."""
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "id",
                "keep_1_or_0",
                "issue_created",
                "question",
                "reference_answer_preview",
                "has_version",
                "has_code_block",
                "issue_url",
                "review_notes",
            ]
        )
        for cand in candidates:
            preview = cand["reference_answer"].replace("\n", " ")[:600]
            writer.writerow(
                [
                    cand["id"],
                    "",
                    (cand.get("issue_created_at") or "")[:10],
                    cand["query"],
                    preview,
                    int(cand["features"]["has_version"]),
                    int(cand["features"]["has_code_block"]),
                    cand.get("issue_url") or "",
                    "",
                ]
            )


def finalize(
    candidates_path: Path,
    review_path: Path,
    out_path: Path,
    *,
    auto_keep: int = 0,
) -> int:
    """Apply the human keep/drop column and emit the committed eval set.

    ``auto_keep`` exists only to smoke-test the pipeline before anyone has
    reviewed the sheet. Items taken that way are stamped ``auto_selected: true``
    so an unreviewed eval set can never be mistaken for a reviewed one.
    """
    candidates = load_jsonl(candidates_path)

    if auto_keep > 0:
        kept = candidates[:auto_keep]
        for item in kept:
            item["auto_selected"] = True
        with out_path.open("w", encoding="utf-8") as handle:
            for item in kept:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(
            f"WARNING: wrote {len(kept)} items WITHOUT human review "
            f"(auto_selected=true). Do not publish numbers from this set; "
            f"fill in {review_path.name} and rerun with --finalize."
        )
        return len(kept)

    keep: dict[str, int] = {}
    with review_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            raw = (row.get("keep_1_or_0") or "").strip()
            if raw in {"1", "y", "Y", "true", "True"}:
                keep[row["id"]] = 1
            elif raw in {"0", "n", "N", "false", "False"}:
                keep[row["id"]] = 0

    kept = []
    for cand in candidates:
        decision = keep.get(cand["id"])
        if decision != 1:
            continue
        cand.pop("keep", None)
        kept.append(cand)

    with out_path.open("w", encoding="utf-8") as handle:
        for item in kept:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"kept {len(kept)} / {len(keep)} reviewed -> {out_path}")
    return len(kept)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the evaluation set")
    parser.add_argument("--preset", default="vercel-ai")
    parser.add_argument("--max-candidates", type=int, default=200)
    parser.add_argument(
        "--min-answer-chars",
        type=int,
        default=MIN_ANSWER_CHARS,
        help=(
            "reference-answer floor. Lower it for repositories whose maintainers "
            "reply tersely; short is not the same as uninformative"
        ),
    )
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument(
        "--auto-keep",
        type=int,
        default=0,
        help="take the first N candidates without review, for smoke tests only",
    )
    args = parser.parse_args(argv)

    preset: RepoPreset = load_preset(args.preset)
    candidates_path = preset.processed_dir / CANDIDATES_FILE
    review_path = preset.processed_dir / REVIEW_FILE
    evalset_path = preset.processed_dir / EVALSET_FILE

    if args.finalize:
        finalize(candidates_path, review_path, evalset_path, auto_keep=args.auto_keep)
        return 0

    issues = load_jsonl(preset.raw_dir / ISSUES_FILE)
    comments = load_jsonl(preset.raw_dir / COMMENTS_FILE)
    funnel: dict[str, int] = {}
    candidates = collect_candidates(
        issues,
        comments,
        min_answer_chars=args.min_answer_chars,
        stats=funnel,
    )[: args.max_candidates]

    with candidates_path.open("w", encoding="utf-8") as handle:
        for cand in candidates:
            handle.write(json.dumps(cand, ensure_ascii=False) + "\n")
    write_review_sheet(candidates, review_path)

    feats = Counter()
    for cand in candidates:
        for key in ("has_version", "has_file_path", "has_code_block", "has_symbol"):
            feats[key] += int(cand["features"][key])

    print(f"candidates: {len(candidates)}")
    print(f"  with version string : {feats['has_version']}")
    print(f"  with file path      : {feats['has_file_path']}")
    print(f"  with code block     : {feats['has_code_block']}")
    if funnel:
        print("\nmining funnel (every stage that discards data):")
        print(f"  comments harvested          {funnel.get('comments_total', 0):>6}")
        print(f"  - automated accounts        {funnel.get('comments_total', 0) - funnel.get('comments_human', 0):>6}")
        print(f"  = human comments            {funnel.get('comments_human', 0):>6}")
        print(f"    of which maintainer       {funnel.get('comments_maintainer', 0):>6}")
        print(f"  issues in scope             {funnel.get('issues_in_scope', 0):>6}")
        print(f"  - no maintainer reply       {funnel.get('issues_in_scope', 0) - funnel.get('issues_with_maintainer', 0):>6}")
        print(f"  = with maintainer reply     {funnel.get('issues_with_maintainer', 0):>6}")
        print(f"  - answer too short          {funnel.get('comments_rejected_short', 0):>6}")
        print(f"  - triage boilerplate        {funnel.get('comments_rejected_triage', 0):>6}")
        print(f"  = usable question/answer    {funnel.get('issues_with_usable_answer', 0):>6}")
        print(f"  - duplicate titles          {funnel.get('issues_with_usable_answer', 0) - len(candidates):>6}")
        print(f"  = candidates                {len(candidates):>6}")
    print(f"-> {candidates_path}")
    print(f"-> {review_path}   (fill keep_1_or_0, then rerun with --finalize)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
