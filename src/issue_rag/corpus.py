"""把抓到的 issue 与评论清洗、去机器人、按评论分块成检索语料。

Turn harvested issues + comments into a retrieval corpus.

Retrieval unit
--------------
One document per *issue body* and one per *comment*, both prefixed with the
issue title. That choice matters more than the embedding model:

* An issue comment is usually a self-contained answer ("on v5 you have to call
  ``convertToModelMessages`` first"), so it retrieves and cites well on its own.
* Prefixing every chunk with the issue title matters because chunk-level text
  often loses the subject ("this also happens with streaming").
* Keeping the issue body as its own document lets the system cite the problem
  statement as well as the fix.

Long bodies are chunked on paragraph boundaries with overlap so a fix that sits
at the bottom of a 10 KB bug report is still reachable.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

from .config import RepoPreset, load_preset
from .fetch import COMMENTS_FILE, ISSUES_FILE

CORPUS_FILE = "corpus.jsonl"
STATS_FILE = "corpus_stats.json"

MAINTAINER_ROLES = frozenset({"MEMBER", "OWNER", "COLLABORATOR"})

# Bots post a large share of issue traffic and their comments are useless as
# answers ("This issue is stale", "Thanks for the report!").
BOT_PATTERN = re.compile(r"(\[bot\]$|-bot$|^github-actions$|^vercel$|^dependabot)", re.I)

HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
EXCESS_BLANKS = re.compile(r"\n{3,}")

MAX_CHUNK_CHARS = 1_200
CHUNK_OVERLAP = 150


def clean_text(text: str) -> str:
    """Strip noise that hurts both retrieval and judging."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n")
    text = HTML_COMMENT.sub("", text)
    text = EXCESS_BLANKS.sub("\n\n", text)
    return text.strip()


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """按段落边界分块，尽量塞满 max_chars 并在块间保留 overlap。

    Split on paragraph boundaries, packing paragraphs up to ``max_chars``."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    size = 0

    for para in paragraphs:
        # A single huge paragraph (stack trace, minified JSON) has to be cut.
        while len(para) > max_chars:
            head, para = para[:max_chars], para[max_chars - overlap :]
            if current:
                chunks.append("\n\n".join(current))
                current, size = [], 0
            chunks.append(head)

        if size + len(para) + 2 > max_chars and current:
            chunks.append("\n\n".join(current))
            tail = "\n\n".join(current)
            carry = tail[-overlap:] if overlap else ""
            current = [carry, para] if carry else [para]
            size = sum(len(c) + 2 for c in current)
        else:
            current.append(para)
            size += len(para) + 2

    if current:
        chunks.append("\n\n".join(current))
    return [c.strip() for c in chunks if c.strip()]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def is_bot(login: str | None) -> bool:
    return bool(login and BOT_PATTERN.search(login))


def build_documents(
    issues: Iterable[dict[str, Any]],
    comments: Iterable[dict[str, Any]],
    repo: str,
    *,
    drop_bots: bool = True,
) -> Iterator[dict[str, Any]]:
    """Yield corpus documents, joined on issue number.

    ``drop_bots`` defaults to True because automated accounts dominate recent
    comment traffic in some repositories -- in the repo this was built against,
    two bots accounted for 84% of a 2000-comment sample. Leaving them in both
    pollutes retrieval and makes every question look answered.
    """
    by_number: dict[int, dict[str, Any]] = {}
    for issue in issues:
        if issue.get("is_pull_request"):
            continue
        by_number[issue["number"]] = issue

    def header(issue: dict[str, Any]) -> str:
        labels = ", ".join(issue.get("labels") or [])
        suffix = f" [labels: {labels}]" if labels else ""
        return f"Issue #{issue['number']}: {issue['title']}{suffix}"

    for number, issue in by_number.items():
        body = clean_text(issue.get("body") or "")
        if body:
            for idx, chunk in enumerate(chunk_text(body)):
                yield {
                    "doc_id": f"issue-{number}-body-{idx}",
                    "issue_number": number,
                    "kind": "issue_body",
                    "title": issue["title"],
                    "author": issue.get("author"),
                    "author_association": None,
                    "is_maintainer": False,
                    "created_at": issue.get("created_at"),
                    "url": issue.get("html_url"),
                    "text": f"{header(issue)}\n\n{chunk}",
                }

    # Comments are attached after bodies so document order stays stable, which
    # keeps cached embeddings valid across rebuilds.
    grouped: dict[int, list[dict[str, Any]]] = {}
    for comment in comments:
        number = comment.get("issue_number")
        if number not in by_number:
            continue
        if drop_bots and is_bot(comment.get("author")):
            continue
        grouped.setdefault(number, []).append(comment)

    for number, items in grouped.items():
        issue = by_number[number]
        items.sort(key=lambda c: (c.get("created_at") or "", c.get("id") or 0))
        for comment in items:
            login = comment.get("author")
            body = clean_text(comment.get("body") or "")
            if not body:
                continue
            for idx, chunk in enumerate(chunk_text(body)):
                yield {
                    "doc_id": f"issue-{number}-comment-{comment['id']}-{idx}",
                    "issue_number": number,
                    "comment_id": comment["id"],
                    "kind": "issue_comment",
                    "title": issue["title"],
                    "author": login,
                    "author_association": comment.get("author_association"),
                    "is_maintainer": comment.get("author_association") in MAINTAINER_ROLES,
                    "is_bot": is_bot(login),
                    "reactions": comment.get("reactions", 0),
                    "created_at": comment.get("created_at"),
                    "url": comment.get("html_url"),
                    "text": f"{header(issue)}\n\nComment by {login} "
                    f"({comment.get('author_association') or 'N/A'}):\n{chunk}",
                }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the retrieval corpus")
    parser.add_argument("--preset", default="vercel-ai")
    parser.add_argument(
        "--keep-bots",
        action="store_true",
        help="keep comments from automated accounts (off by default)",
    )
    args = parser.parse_args(argv)

    preset: RepoPreset = load_preset(args.preset)
    issues = load_jsonl(preset.raw_dir / ISSUES_FILE)
    comments = load_jsonl(preset.raw_dir / COMMENTS_FILE)
    print(f"loaded {len(issues)} issues, {len(comments)} comments")

    docs = list(
        build_documents(issues, comments, preset.repo, drop_bots=not args.keep_bots)
    )
    out_path = preset.processed_dir / CORPUS_FILE
    with out_path.open("w", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(json.dumps(doc, ensure_ascii=False) + "\n")

    kinds = Counter(d["kind"] for d in docs)
    maintainer = sum(1 for d in docs if d.get("is_maintainer"))
    issues_covered = len({d["issue_number"] for d in docs})
    issues_with_answers = len(
        {d["issue_number"] for d in docs if d["kind"] == "issue_comment"}
    )
    bot_comments = sum(1 for c in comments if is_bot(c.get("author")))
    lengths = sorted(len(d["text"]) for d in docs) or [0]
    stats = {
        "repo": preset.repo,
        "documents": len(docs),
        "by_kind": dict(kinds),
        "maintainer_documents": maintainer,
        "issues_covered": issues_covered,
        "issues_with_answer_comment": issues_with_answers,
        "issues_without_answers": issues_covered - issues_with_answers,
        "comments_harvested": len(comments),
        "bot_comments_dropped": 0 if args.keep_bots else bot_comments,
        "chars_p50": lengths[len(lengths) // 2],
        "chars_p95": lengths[int(len(lengths) * 0.95)],
        "chars_max": lengths[-1],
    }
    (preset.processed_dir / STATS_FILE).write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, indent=2))
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
