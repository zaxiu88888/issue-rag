"""Harvest raw issues and issue comments into JSONL.

Usage::

    python -m issue_rag.fetch issues   --preset vercel-ai --months 12
    python -m issue_rag.fetch comments --preset vercel-ai --months 12
    python -m issue_rag.fetch all      --preset vercel-ai --months 12

Why two stages instead of one:

* Issues come from the *search* endpoint, which has its own quota (10 requests
  per minute anonymous) and returns 100 issues with bodies per page.
* Comments come from the *repo-wide comment list*, ``/repos/{o}/{r}/issues/
  comments``, which returns 100 comments per request. Fetching comments per
  issue would cost one core request each and burn the 60/hour budget instantly.

Both stages append to JSONL and skip records already harvested, so they are safe
to interrupt and rerun.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
from typing import Any, Iterable

from .config import CACHE_DIR, RepoPreset, load_env_file, load_preset
from .github import GitHubClient, GitHubError, QuotaExhausted

ISSUES_FILE = "issues.jsonl"
COMMENTS_FILE = "comments.jsonl"
MANIFEST_FILE = "manifest.json"

# Long bug reports can run to tens of kilobytes; the tail is usually stack
# traces, so a generous cap keeps the corpus focused without losing the report.
MAX_BODY_CHARS = 12_000


def _since_date(months: int) -> str:
    today = dt.date.today()
    # Approximate month arithmetic by 30.44 days; precise enough for a window.
    start = today - dt.timedelta(days=int(months * 30.44))
    return start.isoformat()


def _existing_keys(path: pathlib.Path, key: str) -> set[Any]:
    keys: set[Any] = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                keys.add(json.loads(line)[key])
            except (json.JSONDecodeError, KeyError):
                continue
    return keys


def _append_jsonl(path: pathlib.Path, records: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def _trim(text: str | None) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n")
    if len(text) <= MAX_BODY_CHARS:
        return text
    return text[:MAX_BODY_CHARS] + "\n\n[... truncated by harvester ...]"


# --------------------------------------------------------------------- issues


def harvest_issues(
    client: GitHubClient,
    preset: RepoPreset,
    months: int,
    max_issues: int,
    window_axis: str = "updated",
) -> int:
    """Pull closed issues that have at least one comment (i.e. a possible answer).

    ``window_axis`` matters more than it looks. Comments are harvested from the
    repo-wide comment list, which GitHub returns **sorted by update time**, so
    the issue query has to select on the same axis. Selecting issues by
    ``created`` while pulling comments by ``updated`` produces two samples that
    barely intersect: in one run, 150 harvested comments yielded only 12
    documents, because the rest belonged to issues created before the window.
    """
    out_path = preset.raw_dir / ISSUES_FILE
    seen = _existing_keys(out_path, "number")
    if seen:
        print(f"  resuming: {len(seen)} issues already on disk")

    since = _since_date(months)
    field = "updated" if window_axis == "updated" else "created"
    query = (
        f"repo:{preset.repo} is:issue is:closed comments:>=1 "
        f"{field}:>={since} -label:duplicate"
    )
    print(f"  query: {query}")

    params = {"q": query, "sort": "updated", "order": "desc"}
    records: list[dict[str, Any]] = []

    # 50 per page rather than 100: a 100-item search page is ~800 KB and this
    # network truncates chunked responses at that size.
    for item in client.paginate(
        "/search/issues", params, max_items=max_issues, max_pages=30, per_page=50
    ):
        number = item.get("number")
        if number is None or number in seen:
            continue
        records.append(
            {
                "number": number,
                "title": item.get("title") or "",
                "body": _trim(item.get("body")),
                "state": item.get("state"),
                "created_at": item.get("created_at"),
                "closed_at": item.get("closed_at"),
                "updated_at": item.get("updated_at"),
                "comments_count": item.get("comments", 0),
                "html_url": item.get("html_url"),
                "author": (item.get("user") or {}).get("login"),
                "labels": [lbl.get("name") for lbl in (item.get("labels") or [])],
                "is_pull_request": "pull_request" in item,
            }
        )

    written = _append_jsonl(out_path, records)
    print(f"  wrote {written} issues -> {out_path}")
    _update_manifest(
        preset,
        {"issues": {"since": since, "axis": window_axis, "written": written}},
    )
    return written


# ------------------------------------------------------------------- comments


def _issue_number_from_url(issue_url: str | None) -> int | None:
    if not issue_url:
        return None
    tail = issue_url.rstrip("/").rsplit("/", 1)[-1]
    return int(tail) if tail.isdigit() else None


def harvest_comments(
    client: GitHubClient,
    preset: RepoPreset,
    months: int,
    max_pages: int,
) -> int:
    """Pull repo-wide issue comments, newest first."""
    out_path = preset.raw_dir / COMMENTS_FILE
    seen = _existing_keys(out_path, "id")
    if seen:
        print(f"  resuming: {len(seen)} comments already on disk")

    since = _since_date(months)
    params = {"since": f"{since}T00:00:00Z", "sort": "updated", "direction": "desc"}
    endpoint = f"/repos/{preset.repo}/issues/comments"

    records: list[dict[str, Any]] = []
    # 50 per page: a 100-comment page is ~750 KB and this network drops chunked
    # responses above roughly that size (IncompleteRead), so smaller pages are
    # the pragmatic choice over fighting the transport.
    for item in client.paginate(endpoint, params, max_pages=max_pages, per_page=50):
        cid = item.get("id")
        if cid is None or cid in seen:
            continue
        records.append(
            {
                "id": cid,
                "issue_number": _issue_number_from_url(item.get("issue_url")),
                "body": _trim(item.get("body")),
                "author": (item.get("user") or {}).get("login"),
                "author_association": item.get("author_association"),
                "created_at": item.get("created_at"),
                "html_url": item.get("html_url"),
                "reactions": (item.get("reactions") or {}).get("total_count", 0),
            }
        )

    written = _append_jsonl(out_path, records)
    print(f"  wrote {written} comments -> {out_path}")
    _update_manifest(preset, {"comments": {"since": since, "written": written}})
    return written


def harvest_comments_per_issue(
    client: GitHubClient,
    preset: RepoPreset,
    max_issues: int,
) -> int:
    """Fetch each issue's comments individually.

    Necessary because the repo-wide ``/issues/comments`` endpoint returns
    comments on **pull requests as well as issues** -- in a repo like Vite, most
    of them. Joining that stream against a PR-free issue list drops almost
    everything: one sample of 150 repo-wide comments produced 16 usable
    documents.

    Costs one request per issue, so it is budgeted and resumable: issues are
    visited in descending comment count (a busy thread is likelier to contain a
    maintainer reply), and already-cached pages are free to revisit.
    """
    issues_path = preset.raw_dir / ISSUES_FILE
    issues = [json.loads(line) for line in issues_path.open(encoding="utf-8") if line.strip()]
    issues.sort(key=lambda i: -(i.get("comments_count") or 0))

    out_path = preset.raw_dir / COMMENTS_FILE
    seen = _existing_keys(out_path, "id")
    print(f"  {len(issues)} issues on disk, {len(seen)} comments already harvested")

    written = 0
    with out_path.open("a", encoding="utf-8") as handle:
        for position, issue in enumerate(issues[:max_issues], start=1):
            number = issue["number"]
            endpoint = f"/repos/{preset.repo}/issues/{number}/comments"
            page_records = []
            try:
                for item in client.paginate(endpoint, {}, per_page=50, max_pages=3):
                    cid = item.get("id")
                    if cid is None or cid in seen:
                        continue
                    page_records.append(
                        {
                            "id": cid,
                            "issue_number": number,
                            "body": _trim(item.get("body")),
                            "author": (item.get("user") or {}).get("login"),
                            "author_association": item.get("author_association"),
                            "created_at": item.get("created_at"),
                            "html_url": item.get("html_url"),
                            "reactions": (item.get("reactions") or {}).get("total_count", 0),
                        }
                    )
            except (QuotaExhausted, GitHubError) as exc:
                print(f"  stopping at issue {number}: {exc}")
                break

            for record in page_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                seen.add(record["id"])
            handle.flush()
            written += len(page_records)
            maintainers = sum(
                1
                for r in page_records
                if r["author_association"] in {"MEMBER", "OWNER", "COLLABORATOR"}
            )
            print(
                f"  {position:>3}/{min(max_issues, len(issues))} #{number:<6} "
                f"+{len(page_records):2d} comments ({maintainers} maintainer)"
            )

    print(f"  wrote {written} comments -> {out_path}")
    _update_manifest(preset, {"comments": {"mode": "per-issue", "written": written}})
    return written


# ------------------------------------------------------------------- manifest


def _update_manifest(preset: RepoPreset, patch: dict[str, Any]) -> None:
    path = preset.raw_dir / MANIFEST_FILE
    manifest: dict[str, Any] = {}
    if path.exists():
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
    manifest.setdefault("repo", preset.repo)
    manifest["updated_at"] = dt.datetime.now(tz=dt.timezone.utc).isoformat()
    for key, value in patch.items():
        manifest.setdefault(key, {}).update(value)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


# ------------------------------------------------------------------------ cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("issues", "comments", "all"))
    parser.add_argument("--preset", default="vercel-ai")
    parser.add_argument("--months", type=int, default=12, help="recency window")
    parser.add_argument(
        "--window-axis",
        choices=("updated", "created"),
        default="updated",
        help=(
            "which timestamp selects issues. Keep the default: comments are "
            "returned sorted by update time, so selecting issues by creation "
            "date yields two barely-overlapping samples"
        ),
    )
    parser.add_argument("--max-issues", type=int, default=800)
    parser.add_argument("--max-pages", type=int, default=60, help="comment pages (100/page)")
    parser.add_argument(
        "--mode",
        choices=("repo-wide", "per-issue"),
        default="repo-wide",
        help=(
            "per-issue costs one request per issue but does not waste them on "
            "pull-request comments, which dominate the repo-wide stream"
        ),
    )
    parser.add_argument(
        "--max-comment-issues",
        type=int,
        default=60,
        help="issue cap for --mode per-issue (one request each)",
    )
    parser.add_argument(
        "--allow-wait",
        action="store_true",
        help="sleep through search rate-limit resets instead of failing",
    )
    parser.add_argument("--max-wait-s", type=int, default=120)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # ``GITHUB_TOKEN`` lives in .env (gitignored) like the other credentials, but
    # fetching was the one stage that never read that file -- so a token sitting
    # in .env was silently ignored and every harvest ran on the 60/hour
    # anonymous budget. Load it here, after argument validation, before any I/O.
    loaded = load_env_file()
    if loaded:
        print(f"loaded from .env: {', '.join(loaded)}")
    preset = load_preset(args.preset)
    # Without a cache dir the harvester cannot resume, and a 60/hour quota makes
    # an unresumable harvest close to unusable.
    client = GitHubClient(
        cache_dir=CACHE_DIR, allow_wait=args.allow_wait, max_wait_s=args.max_wait_s
    )
    print(f"target: {preset.repo} ({preset.display_name})")
    print(f"quota:  {client.describe_quota()}")

    try:
        if args.stage in ("issues", "all"):
            print("stage 1/2: issues")
            harvest_issues(
                client, preset, args.months, args.max_issues, args.window_axis
            )
        if args.stage in ("comments", "all"):
            print("stage 2/2: comments")
            if args.mode == "per-issue":
                harvest_comments_per_issue(client, preset, args.max_comment_issues)
            else:
                harvest_comments(client, preset, args.months, args.max_pages)
    except (QuotaExhausted, GitHubError) as exc:
        print(f"\nstopped: {exc}")
        print(f"requests={client.stats.requests} cache_hits={client.stats.cache_hits}")
        return 2

    print(f"\ndone. requests={client.stats.requests} cache_hits={client.stats.cache_hits}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
