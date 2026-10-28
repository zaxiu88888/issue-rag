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
from .judge import Judge
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
    """Recall@k and MRR of the reference issue, with and without self-retrieval."""
    hit_with = 0
    hit_without = 0
    rr_with = 0.0
    rr_without = 0.0
    counted = 0

    for item in items:
        question = item["query"]
        target = item["issue_number"]
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

    n = max(counted, 1)
    return {
        f"recall@{top_k}": round(hit_with / n, 4),
        f"recall@{top_k}_loo": round(hit_without / n, 4),
        "mrr": round(rr_with / n, 4),
        "mrr_loo": round(rr_without / n, 4),
    }


# -------------------------------------------------------------- answer quality


def shard_path(preset: RepoPreset, system: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / f"{preset.slug}__{system}.jsonl"


def load_shard(path: Path) -> dict[str, dict[str, Any]]:
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

    shard = shard_path(preset, system)
    done = load_shard(shard)
    valid_ids = {item["id"] for item in items}
    already = len(valid_ids & set(done))
    print(f"\n[{system}] resuming with {already}/{len(items)} items done")

    with shard.open("a", encoding="utf-8") as handle:
        for position, item in enumerate(items, start=1):
            if item["id"] in done:
                continue
            started = time.time()
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
                    "latency_s": round(time.time() - started, 2),
                    "wall_s": round(time.time() - started, 2),
                    "error": str(exc)[:300],
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"  {position:>3}/{len(items)} {item['id']:<10} "
                    f"{'ERROR':<9} {time.time() - started:5.1f}s  {str(exc)[:60]!r}"
                )
                continue

            record = {
                "item_id": item["id"],
                "system": system,
                "answer": generation.text,
                "verdict": grade.verdict,
                "score": grade.score,
                "fabricated": grade.fabricated,
                "judge_reason": grade.reason,
                "cited": answered_with_citation(generation.text),
                "context_doc_ids": [h.doc_id for h in hits],
                "context_issues": [h.issue_number for h in hits],
                "generated_tokens": generation.output_tokens,
                "latency_s": round(generation.latency_s, 2),
                "wall_s": round(time.time() - started, 2),
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


def write_report(
    preset: RepoPreset,
    summaries: list[dict[str, Any]],
    retrieval: dict[str, dict[str, float]],
    args: argparse.Namespace,
    n_items: int,
    subgroup: list[dict[str, Any]] | None = None,
    n_subgroup: int = 0,
) -> Path:
    lines: list[str] = []
    lines.append(f"# Results — {preset.display_name} (`{preset.repo}`)\n")
    lines.append(f"- evaluation items: **{n_items}**")
    generator = getattr(args, "resolved_model", None) or args.model or args.local_model
    lines.append(f"- generator: `{generator}`")
    lines.append(f"- judge: `{generator}` (same model, greedy decoding)")
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
        lines.append(f"## Retrieval (reference issue found in top-{args.top_k})\n")
        lines.append(
            f"| retriever | recall@{args.top_k} | recall@{args.top_k} leave-one-out | MRR | MRR LOO |"
        )
        lines.append("|---|---|---|---|---|")
        for name, metrics in retrieval.items():
            lines.append(
                f"| `{name}` | {metrics[f'recall@{args.top_k}']:.3f} | "
                f"{metrics[f'recall@{args.top_k}_loo']:.3f} | {metrics['mrr']:.3f} | "
                f"{metrics['mrr_loo']:.3f} |"
            )
        lines.append("")
        lines.append(
            "_leave-one-out hides the source issue's own documents, so it measures "
            "whether the retriever generalises rather than matching the query "
            "against itself._\n"
        )

    path = RESULTS_DIR / f"{preset.slug}__results.md"
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
    parser.add_argument("--retrieval-only", action="store_true")
    return parser


def make_model(args: argparse.Namespace) -> ChatModel:
    return resolve_model(
        provider=args.provider,
        endpoint=args.endpoint,
        model=args.model,
        local_default=args.local_model,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
    if not args.retrieval_only:
        model = make_model(args)
        args.resolved_model = model.name
        judge = Judge(model)
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
    (RESULTS_DIR / f"{preset.slug}__results.json").write_text(
        json.dumps(
            {
                "summaries": summaries,
                "retrieval": retrieval,
                "n_items": len(items),
                "n_closed_book_missed": n_subgroup,
                "subgroup_closed_book_missed": subgroup,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
