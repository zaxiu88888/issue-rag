"""Run the experiment matrix and emit the results table.

Every row of the output answers one question an interviewer would ask:

===================  =========================================================
closed_book          How much does the model already know? (the control)
bm25                 Do we even need embeddings?
dense                Does lexical matching alone explain the gains?
hybrid               Does fusion beat both?
hybrid_rerank        Is a cross-encoder worth its latency?
===================  =========================================================

Retrieval is scored twice: *with* the source issue in the index (the product
number -- a user asking this question should be shown that thread) and
*leave-one-issue-out* (the generalisation number). Both appear in the report.

Runs are resumable: each (system, item) result is appended to a JSONL shard and
replayed on restart, which matters because a local 3B model needs hours for a
full sweep.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from .config import RESULTS_DIR, RepoPreset, load_env_file, load_preset
from .corpus import load_jsonl
from .evalset import EVALSET_FILE
from .generate import ChatModel, GenerationError, resolve_model
from .judge import HolisticJudge, Judge
from .retrieval import Corpus, build_retriever

CORPUS_FILE = "corpus.jsonl"
DEFAULT_SYSTEMS = ("closed_book", "bm25", "dense", "hybrid", "hybrid_rerank")

RAG_SYSTEM_PROMPT = (
    "You are a support engineer for {repo}. Answer the user's question using "
    "ONLY the numbered issue excerpts below.\n\n"
    "Rules:\n"
    "1. Answer the question itself. Do NOT reply with a summary of the thread "
    "(\"this is tracked in issue #123\" is not an answer) unless the thread's "
    "only content is a link to another thread.\n"
    "2. Cite the excerpts you rely on with bracketed numbers, e.g. [1].\n"
    "3. If an excerpt names a version, file, or API, prefer that exact wording.\n"
    "4. Do NOT guess a version number, file path, or API name that is not in the "
    "excerpts. If the excerpts do not answer the question, say exactly what is "
    "missing instead of inventing an answer.\n"
    "5. Be concise: at most six sentences plus a short code snippet if needed."
)

CLOSED_BOOK_SYSTEM_PROMPT = (
    "You are a support engineer for {repo}. Answer the user's question from your "
    "own knowledge. Be concise: at most six sentences plus a short code snippet "
    "if needed. If you are unsure, say so."
)


def build_context(hits: list[Any], max_chars: int = 6_000) -> str:
    """Numbered excerpts, truncated to keep the prompt inside the context window."""
    blocks: list[str] = []
    used = 0
    for idx, hit in enumerate(hits, start=1):
        body = hit.text.strip()
        block = f"[{idx}] {body}"
        if used + len(block) > max_chars:
            remaining = max_chars - used
            if remaining < 200:
                break
            block = block[:remaining] + " [...]"
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def answered_with_citation(answer: str) -> bool:
    import re

    return bool(re.search(r"\[\d+\]", answer or ""))


# ------------------------------------------------------------------ retrieval


def evaluate_retrieval(
    retriever: Any, items: list[dict[str, Any]], top_k: int
) -> dict[str, float]:
    """Retrieval quality on two different questions, because they differ.

    The original metrics asked "did we find the *issue* the question came from".
    On this corpus that is close to a tautology: the query is the issue title and
    every chunk is prefixed with the title, so the gold issue is retrieved at rank
    1 for every item and every retriever -- measured 1.000 recall@5 and 1.000 MRR
    for all four retrievers, on 29 items and on 95 items alike. The
    leave-one-issue-out variant is worse than useless: the reference document is a
    chunk of the gold issue, so removing that issue from the index removes the only
    relevant document *by construction* and the score is identically 0.

    So the metrics that actually discriminate are the ones about the reference
    *comment*:

    * ``gold_chunk@{k}`` -- is the maintainer comment that answers the question
      among the top-k chunks (a chunk-level hit, not an issue-level one)?
    * ``gold_chunk_mrr`` / ``gold_chunk_top1`` -- how high does it rank?
    * ``self_share@{k}`` -- what fraction of the retrieved window comes from the
      gold issue itself? Near 1.0 means the retriever filled the window with
      chunks of the same thread, i.e. the window carries almost no cross-thread
      discrimination. This is the number that explains the saturation above.

    The old issue-level keys are still returned so archived runs stay comparable.
    """
    hit_with = 0
    hit_without = 0
    rr_with = 0.0
    rr_without = 0.0
    gold_hit = 0
    gold_rr = 0.0
    gold_top1 = 0
    self_share = 0.0
    counted = 0

    for item in items:
        question = item["query"]
        target = item["issue_number"]
        gold_comments = {str(cid) for cid in item.get("reference_comment_ids") or []}
        hits_with = retriever.search(question, top_k=top_k)
        hits_without = retriever.search(question, top_k=top_k, exclude_issue=target)
        counted += 1

        ranks_with = [i + 1 for i, h in enumerate(hits_with) if h.issue_number == target]
        ranks_without = [i + 1 for i, h in enumerate(hits_without) if h.issue_number == target]
        if ranks_with:
            hit_with += 1
            rr_with += 1.0 / ranks_with[0]
        if ranks_without:
            hit_without += 1
            rr_without += 1.0 / ranks_without[0]

        if gold_comments:
            gold_ranks = [
                i + 1
                for i, h in enumerate(hits_with)
                if str((h.metadata or {}).get("comment_id")) in gold_comments
            ]
            if gold_ranks:
                gold_hit += 1
                gold_rr += 1.0 / gold_ranks[0]
                gold_top1 += 1 if gold_ranks[0] == 1 else 0
        if hits_with:
            self_share += sum(1 for h in hits_with if h.issue_number == target) / len(hits_with)

    n = max(counted, 1)
    return {
        f"recall@{top_k}": round(hit_with / n, 4),
        f"recall@{top_k}_loo": round(hit_without / n, 4),
        "mrr": round(rr_with / n, 4),
        "mrr_loo": round(rr_without / n, 4),
        f"gold_chunk@{top_k}": round(gold_hit / n, 4),
        "gold_chunk_mrr": round(gold_rr / n, 4),
        "gold_chunk_top1": round(gold_top1 / n, 4),
        f"self_share@{top_k}": round(self_share / n, 4),
    }


# -------------------------------------------------------------- answer quality


def shard_path(preset: RepoPreset, system: str, tag: str = "") -> Path:
    """Per-(system, run-tag) result shard.

    The tag exists so a second run with a different grader lands beside the first
    instead of colliding with it. Re-grading the same answers under two judges is
    what makes the sensitivity analysis apples-to-apples, so the two runs must not
    share a cache key.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"__{tag}" if tag else ""
    return RESULTS_DIR / f"{preset.slug}__{system}{suffix}.jsonl"


def results_stem(preset: RepoPreset, tag: str = "") -> str:
    return f"{preset.slug}__{tag}__results" if tag else f"{preset.slug}__results"


def load_shard(path: Path) -> dict[str, dict[str, Any]]:
    """Last record per item id, *including* failed ones.

    Failures are kept here on purpose: ``summarise`` counts them and the report
    prints an ``errors`` column. Dropping them at load time would quietly raise
    every accuracy figure. Use :func:`finished_ids` to ask which items still
    need to run.
    """
    done: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                done[record["item_id"]] = record
            except (json.JSONDecodeError, KeyError):
                continue
    return done


# Two record shapes are not verdicts and must not count as "done":
#   * ``error``    -- the API call itself failed (e.g. SSL truncation)
#   * ``unparsed`` -- the grader emitted no parseable JSON (budget exhausted)
# Caching either as done makes resume skip the item forever, so n silently
# stops short of the eval-set size and only the errors column would hint at it.
INCOMPLETE_VERDICTS = frozenset({"error", "unparsed"})


def finished_ids(records: dict[str, dict[str, Any]]) -> set[str]:
    """Item ids whose last record is a real verdict, i.e. safe to skip."""
    return {
        item_id
        for item_id, record in records.items()
        if record.get("verdict") not in INCOMPLETE_VERDICTS and not record.get("error")
    }


def run_system(
    system: str,
    items: list[dict[str, Any]],
    corpus: Corpus,
    *,
    model: ChatModel,
    judge: Judge,
    preset: RepoPreset,
    args: argparse.Namespace,
) -> dict[str, Any]:
    retriever = None
    if system != "closed_book":
        mode = "hybrid_rerank" if system == "hybrid_rerank" else system
        retriever = build_retriever(
            mode,
            corpus,
            dense_model=args.dense_model,
            reranker_model=args.reranker_model,
            device=args.device,
        )

    shard = shard_path(preset, system, getattr(args, "run_tag", ""))
    done = load_shard(shard)
    finished = finished_ids(done)
    valid_ids = {item["id"] for item in items}
    already = len(valid_ids & finished)
    failed = len(valid_ids & set(done) - finished)
    print(f"\n[{system}] resuming with {already}/{len(items)} items done")
    if failed:
        print(f"  {failed} recorded failure(s) will be retried, not skipped")

    with shard.open("a", encoding="utf-8") as handle:
        for position, item in enumerate(items, start=1):
            if item["id"] in finished:
                continue
            started = time.perf_counter()
            hits = retriever.search(item["query"], top_k=args.top_k) if retriever else []
            context = build_context(hits) if hits else ""

            if retriever:
                system_prompt = RAG_SYSTEM_PROMPT.format(repo=preset.repo)
                user_content = (
                    f"EXCERPTS:\n{context}\n\nQUESTION:\n{item['query']}"
                    if context
                    else f"QUESTION:\n{item['query']}"
                )
            else:
                system_prompt = CLOSED_BOOK_SYSTEM_PROMPT.format(repo=preset.repo)
                user_content = item["query"]

            try:
                generation = model.chat(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    max_new_tokens=args.max_new_tokens,
                )
                grade = judge.grade(
                    item["query"], item["reference_answer"], generation.text
                )
            except GenerationError as exc:
                # A failed call is not a wrong answer. Record it, keep the run
                # alive, and let the report show an error rate alongside n --
                # silently dropping it would quietly improve every metric.
                record = {
                    "item_id": item["id"],
                    "system": system,
                    "answer": "",
                    "verdict": "error",
                    "score": 0.0,
                    "fabricated": False,
                    "judge_reason": "",
                    "cited": False,
                    "context_doc_ids": [h.doc_id for h in hits],
                    "context_issues": [h.issue_number for h in hits],
                    "generated_tokens": 0,
                    "latency_s": round(time.perf_counter() - started, 2),
                    "wall_s": round(time.perf_counter() - started, 2),
                    "error": str(exc)[:300],
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"  {position:>3}/{len(items)} {item['id']:<10} "
                    f"{'ERROR':<9} {time.perf_counter() - started:5.1f}s  {str(exc)[:60]!r}"
                )
                continue

            record = {
                "item_id": item["id"],
                "system": system,
                "answer": generation.text,
                "verdict": grade.verdict,
                "score": grade.score,
                "fabricated": grade.fabricated,
                # The four sub-answers that produced the verdict. Storing them
                # makes a verdict auditable ("why partial?") instead of opaque,
                # which is what a human reviewer needs during spot checks.
                "checks": grade.checks,
                "judge_reason": grade.reason,
                "cited": answered_with_citation(generation.text),
                "context_doc_ids": [h.doc_id for h in hits],
                "context_issues": [h.issue_number for h in hits],
                "generated_tokens": generation.output_tokens,
                "latency_s": round(generation.latency_s, 2),
                "wall_s": round(time.perf_counter() - started, 2),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"  {position:>3}/{len(items)} {item['id']:<10} "
                f"{grade.verdict:<9} {generation.latency_s:5.1f}s  {generation.text[:60]!r}"
            )

    # Summarise only the items in the current eval set. A shard is a cache keyed
    # by item id, so it can still hold results for items that have since been
    # dropped during review; counting those silently inflates n and drags every
    # metric with them.
    valid_ids = {item["id"] for item in items}
    return summarise(
        (rec for iid, rec in load_shard(shard).items() if iid in valid_ids), system
    )


def summarise(records: Any, system: str) -> dict[str, Any]:
    records = list(records)
    errors = [r for r in records if r.get("error")]
    records = [r for r in records if not r.get("error")]
    n = len(records)
    if not n:
        return {"system": system, "n": 0, "errors": len(errors)}
    verdicts = [r["verdict"] for r in records]
    latencies = sorted(r["latency_s"] for r in records)

    def pct(value: float) -> float:
        return round(100.0 * value, 1)

    return {
        "system": system,
        "n": n,
        "errors": len(errors),
        "error_rate": pct(len(errors) / max(len(records) + len(errors), 1)),
        "accuracy": round(statistics.mean(r["score"] for r in records), 4),
        "correct_rate": pct(verdicts.count("correct") / n),
        "partial_rate": pct(verdicts.count("partial") / n),
        "incorrect_rate": pct(verdicts.count("incorrect") / n),
        "unparsed_rate": pct(verdicts.count("unparsed") / n),
        "fabricated_rate": pct(sum(1 for r in records if r["fabricated"]) / n),
        "cited_rate": pct(sum(1 for r in records if r["cited"]) / n),
        "latency_p50_s": round(latencies[n // 2], 2),
        "latency_p95_s": round(latencies[min(n - 1, int(n * 0.95))], 2),
    }


# --------------------------------------------------------------------- report


def regrade_system(
    system: str,
    items: list[dict[str, Any]],
    *,
    judge: Judge,
    preset: RepoPreset,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Re-grade existing answers with a different judge, without re-generating.

    Reusing the stored answers is the whole point: it holds the generator fixed so
    the only thing that changes between the two runs is who grades. Regenerating
    would make the comparison confounded by sampling differences and would cost
    twice as much.
    """
    base = load_shard(shard_path(preset, system, args.regrade_from))
    target = shard_path(preset, system, args.run_tag)
    # Same rule as the generation path: only a real verdict counts as done.
    # "unparsed" means the grader emitted no parseable JSON -- usually a reasoning
    # model whose trace consumed the whole budget -- and "error" means the call
    # failed outright. Neither is a verdict, so a rerun should retry them.
    # Re-appending is safe because load_shard keys by item_id and keeps the last
    # record, so a later success supersedes the earlier failure.
    finished = finished_ids(load_shard(target))
    source_label = args.regrade_from or "untagged"
    print(
        f"\n[{system}] re-grading from '{source_label}' with {judge.model.name} "
        f"(answers reused)"
    )

    with target.open("a", encoding="utf-8") as handle:
        for position, item in enumerate(items, start=1):
            if item["id"] in finished:
                continue
            source = base.get(item["id"])
            if source is None or source.get("error"):
                print(f"  {position:>3}/{len(items)} {item['id']:<10} no baseline answer, skipped")
                continue
            try:
                grade = judge.grade(
                    item["query"], item["reference_answer"], source["answer"]
                )
            except GenerationError as exc:
                record = {
                    **source,
                    "verdict": "error",
                    "score": 0.0,
                    "error": str(exc)[:300],
                    "judge_model": judge.model.name,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"  {position:>3}/{len(items)} {item['id']:<10} ERROR {str(exc)[:50]!r}")
                continue

            record = {
                **source,
                "verdict": grade.verdict,
                "score": grade.score,
                "fabricated": grade.fabricated,
                "checks": grade.checks,
                "judge_reason": grade.reason,
                "judge_model": judge.model.name,
                "regraded_from": source.get("judge_model") or "generator",
            }
            record.pop("error", None)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"  {position:>3}/{len(items)} {item['id']:<10} "
                f"{grade.verdict:<9} (was {source.get('verdict')})"
            )

    valid_ids = {item["id"] for item in items}
    return summarise(
        (rec for iid, rec in load_shard(target).items() if iid in valid_ids), system
    )


def write_report(
    preset: RepoPreset,
    summaries: list[dict[str, Any]],
    retrieval: dict[str, dict[str, float]],
    args: argparse.Namespace,
    n_items: int,
    subgroup: list[dict[str, Any]] | None = None,
    n_subgroup: int = 0,
) -> Path | None:
    # A run that produced neither an answer table nor a retrieval table has
    # nothing to say. Writing anyway would replace a good report with an empty
    # one, which is what `--retrieval-only --systems closed_book` would do.
    if not any(row.get("n") for row in summaries) and not retrieval:
        print("no results to report; leaving the existing results.md untouched")
        return None
    lines: list[str] = []
    lines.append(f"# Results — {preset.display_name} (`{preset.repo}`)\n")
    lines.append(f"- evaluation items: **{n_items}**")
    generator = getattr(args, "resolved_model", None) or args.model or args.local_model
    judge_name = getattr(args, "resolved_judge", None) or generator
    if getattr(args, "retrieval_only", False):
        # No model was called, so printing one would be a fabricated provenance
        # line -- exactly the kind of quiet lie the rest of this repo guards against.
        lines.append("- generator: _not used (retrieval-only run)_")
        lines.append("- judge: _not used (retrieval-only run)_")
    else:
        lines.append(f"- generator: `{generator}`")
        same = " (same model, greedy decoding)" if judge_name == generator else ""
        lines.append(f"- judge: `{judge_name}`{same}")
    lines.append(f"- top-k retrieved: {args.top_k}")
    lines.append("")

    lines.append("## Answer quality\n")
    header = (
        "| system | n | errors | accuracy | correct | partial | incorrect | fabricated | "
        "cited | p50 s | p95 s |"
    )
    lines.append(header)
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for row in summaries:
        if not row.get("n"):
            continue
        lines.append(
            f"| `{row['system']}` | {row['n']} | {row.get('errors', 0)} | "
            f"{row['accuracy']:.3f} | {row['correct_rate']}% | "
            f"{row['partial_rate']}% | {row['incorrect_rate']}% | "
            f"{row['fabricated_rate']}% | {row['cited_rate']}% | "
            f"{row['latency_p50_s']} | {row['latency_p95_s']} |"
        )
    lines.append("")
    lines.append(
        "_accuracy = correct(1.0) + partial(0.5) averaged; fabricated = judge "
        "flagged an invented version/API/path; failed calls are counted in "
        "`errors` and excluded from accuracy._\n"
    )

    if subgroup:
        lines.append(
            f"## Where retrieval actually matters (n={n_subgroup})\n"
        )
        lines.append(
            "Restricted to the items the closed-book model got **wrong**. On items "
            "the model already answers from memory, retrieval has nothing to add, "
            "and including them dilutes the comparison. This is a subgroup of the "
            "same run, not a filtered eval set: the closed-book row above still "
            "counts every item, so the full table is not cherry-picked.\n"
        )
        lines.append("| system | accuracy | correct | partial | incorrect | fabricated |")
        lines.append("|---|---|---|---|---|---|")
        for row in subgroup:
            if not row.get("n"):
                continue
            lines.append(
                f"| `{row['system']}` | {row['accuracy']:.3f} | {row['correct_rate']}% | "
                f"{row['partial_rate']}% | {row['incorrect_rate']}% | "
                f"{row['fabricated_rate']}% |"
            )
        lines.append("")

    if retrieval:
        k = args.top_k
        lines.append(f"## Retrieval (the maintainer's own comment found in top-{k})\n")
        lines.append(
            "Relevance here is the reference **comment**, not the issue it came "
            "from. The issue-level version of this table is in Limitations "
            "because it is a tautology on this corpus.\n"
        )
        lines.append(
            f"| retriever | gold chunk@{k} | gold chunk MRR | gold chunk top-1 | "
            f"share of window from the same issue |"
        )
        lines.append("|---|---|---|---|---|")
        for name, metrics in retrieval.items():
            lines.append(
                f"| `{name}` | {metrics[f'gold_chunk@{k}']:.3f} | "
                f"{metrics['gold_chunk_mrr']:.3f} | {metrics['gold_chunk_top1']:.3f} | "
                f"{metrics[f'self_share@{k}']:.3f} |"
            )
        lines.append("")
        lines.append(
            f"_gold chunk@{k} = the maintainer comment that answers the question "
            "appears among the retrieved chunks. The last column is the diagnosis "
            "of why the issue-level metric saturates: almost the whole window is "
            "chunks of the gold issue itself, so `recall@k` and MRR are 1.000 for "
            "every retriever by construction._\n"
        )
        lines.append(
            f"_Archived issue-level numbers (recall@{k} / leave-one-out / MRR), "
            "kept only to show the degradation: "
            + " · ".join(
                f"`{name}` {m[f'recall@{k}']:.3f}/{m[f'recall@{k}_loo']:.3f}/{m['mrr']:.3f}"
                for name, m in retrieval.items()
            )
            + "._\n"
        )

    path = RESULTS_DIR / f"{results_stem(preset, getattr(args, 'run_tag', ''))}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ------------------------------------------------------------------------ cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the evaluation matrix")
    parser.add_argument("--preset", default="vercel-ai")
    parser.add_argument("--systems", default=",".join(DEFAULT_SYSTEMS))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="0 = all eval items")
    parser.add_argument(
        "--provider",
        default="local",
        choices=("local", "deepseek", "openai", "siliconflow", "ollama"),
        help="serving backend; 'local' loads weights from disk, the others call an API",
    )
    parser.add_argument("--model", default="", help="override the provider's default model")
    parser.add_argument(
        "--local-model",
        default="data/models/Qwen2.5-3B-Instruct",
        help="weights path used when --provider local",
    )
    parser.add_argument("--dense-model", default="data/models/bge-base-en-v1.5")
    parser.add_argument("--reranker-model", default="data/models/bge-reranker-base")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument(
        "--endpoint",
        default="",
        help="explicit OpenAI-compatible base URL; overrides --provider's URL",
    )
    parser.add_argument(
        "--judge-provider",
        default=None,
        help="provider for the grader; defaults to --provider (see make_judge)",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help=(
            "model used to grade; defaults to the generator. Point this at a "
            "different family to remove the self-grading circularity"
        ),
    )
    parser.add_argument("--judge-endpoint", default="", help="base URL for the grader")
    parser.add_argument(
        "--judge-rubric",
        choices=("checks", "holistic"),
        default="checks",
        help=(
            "which rubric the grader uses. 'checks' (default) is the shipped four "
            "questions with the verdict mapped in code; 'holistic' is the original "
            "single-verdict prompt, kept only so the rubric A/B can be regenerated "
            "by a command. Never publish numbers produced with 'holistic'"
        ),
    )
    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        default=160,
        help=(
            "token budget for the grader. Reasoning models spend most of a small "
            "budget on their trace and return an EMPTY final message, so raise "
            "this to ~3000 when --judge-model is a reasoning model"
        ),
    )
    parser.add_argument(
        "--run-tag",
        default="",
        help=(
            "suffix for shard and report filenames, so several runs can coexist. "
            "Required with --regrade."
        ),
    )
    parser.add_argument(
        "--regrade",
        action="store_true",
        help=(
            "re-grade the answers already stored in a shard instead of "
            "generating new ones. Holds the generator fixed and varies only "
            "the judge (or the rubric), which is what makes a grader-sensitivity "
            "comparison meaningful."
        ),
    )
    parser.add_argument(
        "--regrade-from",
        default="",
        help=(
            "source arm to re-grade (default: the untagged run). Use this to "
            "re-judge an archived arm into a new tag, e.g. to compare rubrics "
            "over identical answers."
        ),
    )
    parser.add_argument("--retrieval-only", action="store_true")
    return parser


def make_model(args: argparse.Namespace) -> ChatModel:
    return resolve_model(
        provider=args.provider,
        endpoint=args.endpoint,
        model=args.model,
        local_default=args.local_model,
    )


def make_judge(args: argparse.Namespace, generator: ChatModel | None) -> Judge:
    """Build the grader, defaulting to the generator.

    Grading with the same model that generated the answer is circular, and that
    is the project's most-criticisable weakness. Making the judge a flag turns it
    into a supported workflow: point ``--judge-model`` at a different family and
    the circularity is gone at the cost of one more endpoint.

    When the two configurations are identical the generator instance is reused,
    because loading a local model twice would double the memory for nothing.
    """
    judge_provider = args.judge_provider or args.provider
    same_as_generator = (
        judge_provider == args.provider
        and (args.judge_model or "") == (args.model or "")
        and not args.judge_endpoint
    )
    if same_as_generator and generator is not None:
        judge_model = generator
    else:
        judge_model = resolve_model(
            provider=judge_provider,
            endpoint=args.judge_endpoint,
            model=args.judge_model,
            local_default=args.local_model,
        )
    if getattr(args, "judge_rubric", "checks") == "holistic":
        # Opt-in only, for the rubric A/B. The shipped instrument stays Judge.
        return HolisticJudge(judge_model, max_new_tokens=args.judge_max_tokens)
    return Judge(judge_model, max_new_tokens=args.judge_max_tokens)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Validate arguments before touching the filesystem. These guards used to sit
    # after the corpus load, which meant a fresh clone (no built corpus) raised
    # FileNotFoundError instead of SystemExit — the tests looked hermetic locally
    # only because the build artifacts happened to exist.
    if args.regrade and (args.run_tag or "") == (args.regrade_from or ""):
        raise SystemExit(
            "--regrade must write somewhere other than the arm it reads.\n"
            f"  reading : {args.regrade_from or '(untagged)'}\n"
            f"  writing : {args.run_tag or '(untagged)'}\n"
            "Pick a --run-tag (new arm), or set --regrade-from to an archived arm "
            "you want to re-judge, e.g.:\n"
            "  python -m issue_rag.evaluate --preset vite --regrade \\\n"
            "    --regrade-from v1holistic --run-tag '' "
            "--judge-model deepseek-chat"
        )

    loaded = load_env_file()
    if loaded:
        print(f"loaded from .env: {', '.join(loaded)}")
    preset = load_preset(args.preset)

    evalset_path = preset.processed_dir / EVALSET_FILE
    items = load_jsonl(evalset_path)
    if not items:
        raise SystemExit(
            f"no eval items at {evalset_path}; run issue_rag.evalset, fill in "
            "evalset_review.csv, then rerun evalset with --finalize"
        )
    if args.limit:
        items = items[: args.limit]
    print(f"eval items: {len(items)}")

    corpus = Corpus.load(preset.processed_dir / CORPUS_FILE)
    print(f"corpus docs: {len(corpus)}")

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]

    retrieval: dict[str, dict[str, float]] = {}
    if not args.regrade:
        for system in systems:
            if system == "closed_book":
                continue
            retriever = build_retriever(
                system,
                corpus,
                dense_model=args.dense_model,
                reranker_model=args.reranker_model,
                device=args.device,
            )
            retrieval[system] = evaluate_retrieval(retriever, items, args.top_k)
            print(f"[retrieval:{system}] {retrieval[system]}")

    summaries: list[dict[str, Any]] = []
    subgroup: list[dict[str, Any]] = []
    n_subgroup = 0

    if args.regrade:
        # A re-grade run holds the answers fixed and swaps only the judge, so the
        # two runs are comparable item by item. No retrieval metrics and no
        # subgroup table: both belong to the run that produced the answers, and
        # recomputing them here would mix two judges inside one table.
        judge = make_judge(args, None)
        args.resolved_judge = judge.model.name
        args.resolved_model = "reused from the untagged run"
        for system in systems:
            summaries.append(
                regrade_system(system, items, judge=judge, preset=preset, args=args)
            )
    elif not args.retrieval_only:
        model = make_model(args)
        args.resolved_model = model.name
        judge = make_judge(args, model)
        args.resolved_judge = judge.model.name
        if args.resolved_judge != model.name:
            print(f"judge: {args.resolved_judge} (generator: {model.name})")
        for system in systems:
            summaries.append(
                run_system(
                    system,
                    items,
                    corpus,
                    model=model,
                    judge=judge,
                    preset=preset,
                    args=args,
                )
            )

        # Subgroup analysis: restrict to items the closed-book model got wrong.
        # Reporting this separately avoids the two bad options -- either diluting
        # the comparison with items the model already knows, or filtering the
        # eval set so the closed-book baseline is zero by construction.
        closed_path = shard_path(preset, "closed_book")
        if closed_path.exists():
            valid_ids = {item["id"] for item in items}
            closed = {
                iid: rec for iid, rec in load_shard(closed_path).items() if iid in valid_ids
            }
            hard = {iid for iid, rec in closed.items() if rec.get("score", 0) < 1.0}
            n_subgroup = len(hard)
            for system in systems:
                records = [
                    rec
                    for iid, rec in load_shard(shard_path(preset, system)).items()
                    if iid in hard
                ]
                subgroup.append(summarise(records, system))

    report = write_report(
        preset, summaries, retrieval, args, len(items), subgroup, n_subgroup
    )
    print(f"\n-> {report}")
    stem = results_stem(preset, args.run_tag)
    (RESULTS_DIR / f"{stem}.json").write_text(
        json.dumps(
            {
                "summaries": summaries,
                "retrieval": retrieval,
                "n_items": len(items),
                "n_closed_book_missed": n_subgroup,
                "subgroup_closed_book_missed": subgroup,
                "run_tag": args.run_tag,
                "judge": getattr(args, "resolved_judge", None),
                "regrade": bool(args.regrade),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
