"""Rewrite eval queries the way a user would ask them, and re-measure retrieval.

Why this experiment exists
--------------------------
Every retrieval number in this project uses the **issue title** as the query, and
the Limitations section has always said that favours lexical matching: a title is
close to the words a maintainer used, so BM25 gets an easier case than it would
with a natural question. That was an admitted bias with no measurement behind it.

This module measures it. For each of the 95 eval items it asks a model to rewrite
the title as a question a user would actually type — preserving the technical
facts (versions, API names, error strings) but abandoning the title's wording —
then runs the same four retrievers over both query sets and reports the delta.

What to look for:

* if BM25 drops more than dense, the title advantage was real and the reported
  BM25 numbers are optimistic;
* if all four drop together, the "queries are too easy" worry was overblown;
* if nothing moves, the titles were already question-like.

The paraphrases are cached on disk and keyed by item id, so the run is resumable
and the exact strings used are committed as evidence. The rewrite is deliberately
constrained (see PARAPHRASE_SYSTEM): an unconstrained rewrite can drop the one
detail that makes the question answerable, which would look like a retrieval
failure rather than a query-construction choice.

Usage::

    python -m issue_rag.paraphrase --preset vite            # generate + measure
    python -m issue_rag.paraphrase --preset vite --measure  # reuse, re-measure
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Any

from .config import RESULTS_DIR, RepoPreset, load_env_file, load_preset
from .corpus import load_jsonl
from .evaluate import evaluate_retrieval
from .generate import GenerationError, resolve_model
from .retrieval import Corpus, build_retriever

RETRIEVERS = ("bm25", "dense", "hybrid", "hybrid_rerank")

PARAPHRASE_SYSTEM = """You rewrite bug-report titles as questions a developer \
would actually type when searching for help.

Rules:
1. Keep every technical fact that makes the question answerable: version numbers, \
API names, config keys, error strings, package names, file extensions. Do not \
invent or change them.
2. Drop the title's wording. Do not reuse distinctive phrases from it; a keyword \
search should not be able to match the original title by accident.
3. One or two sentences, plain developer English, no markdown, no quotes.
4. Do not add information that is not implied by the title.

Reply with the rewritten question only, no preamble."""


def paraphrases_path(preset: RepoPreset) -> pathlib.Path:
    return preset.processed_dir / "paraphrases.jsonl"


def load_paraphrases(preset: RepoPreset) -> dict[str, str]:
    path = paraphrases_path(preset)
    if not path.exists():
        return {}
    return {
        row["item_id"]: row["paraphrase"]
        for row in load_jsonl(path)
        if row.get("paraphrase")
    }


def _clean(text: str) -> str:
    """Strip the wrappers models like to add around a one-line answer."""
    line = " ".join(text.strip().split())
    line = re.sub(r'^(question|rewritten question|q)\s*:\s*', "", line, flags=re.I)
    return line.strip('"\' ').strip()


def generate(
    preset: RepoPreset,
    *,
    provider: str,
    model_name: str,
    limit: int = 0,
) -> int:
    items = load_jsonl(preset.processed_dir / "evalset.jsonl")
    if limit:
        items = items[:limit]
    done = load_paraphrases(preset)
    todo = [item for item in items if item["id"] not in done]
    print(f"paraphrases: {len(done)} cached, {len(todo)} to write")
    if not todo:
        return len(done)

    model = resolve_model(provider=provider, model=model_name)
    path = paraphrases_path(preset)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("a", encoding="utf-8") as handle:
        for position, item in enumerate(todo, start=1):
            try:
                result = model.chat(
                    [
                        {"role": "system", "content": PARAPHRASE_SYSTEM},
                        {"role": "user", "content": item["query"]},
                    ],
                    max_new_tokens=160,
                )
            except GenerationError as exc:
                print(f"  {position:>3}/{len(todo)} {item['id']:<9} FAILED {str(exc)[:60]}")
                continue
            paraphrase = _clean(result.text)
            if not paraphrase:
                print(f"  {position:>3}/{len(todo)} {item['id']:<9} EMPTY, skipped")
                continue
            handle.write(
                json.dumps(
                    {"item_id": item["id"], "title": item["query"], "paraphrase": paraphrase},
                    ensure_ascii=False,
                )
                + "\n"
            )
            handle.flush()
            written += 1
            print(f"  {position:>3}/{len(todo)} {item['id']:<9} {paraphrase[:70]}")
    print(f"wrote {written} new paraphrases -> {path}")
    return len(done) + written


def _with_queries(items: list[dict[str, Any]], queries: dict[str, str]) -> list[dict[str, Any]]:
    """Same items, different query strings. Everything else must stay identical."""
    out = []
    for item in items:
        clone = dict(item)
        clone["query"] = queries[item["id"]]
        out.append(clone)
    return out


def measure(
    preset: RepoPreset,
    *,
    top_k: int,
    dense_model: str,
    reranker_model: str,
    device: str | None,
) -> dict[str, Any]:
    items = load_jsonl(preset.processed_dir / "evalset.jsonl")
    paraphrases = load_paraphrases(preset)
    missing = [item["id"] for item in items if item["id"] not in paraphrases]
    if missing:
        raise SystemExit(
            f"{len(missing)} items have no paraphrase ({missing[:3]}...); "
            "run without --measure first"
        )

    corpus = Corpus.load(preset.processed_dir / "corpus.jsonl")
    print(f"corpus: {len(corpus.docs)} documents, {len(items)} items\n")
    paraphrase_items = _with_queries(items, paraphrases)

    report: dict[str, Any] = {"n_items": len(items), "retrieval": {}}
    for mode in RETRIEVERS:
        retriever = build_retriever(
            mode,
            corpus,
            dense_model=dense_model,
            reranker_model=reranker_model if mode == "hybrid_rerank" else None,
            device=device,
        )
        title_metrics = evaluate_retrieval(retriever, items, top_k)
        para_metrics = evaluate_retrieval(retriever, paraphrase_items, top_k)
        report["retrieval"][mode] = {
            "title": title_metrics,
            "paraphrase": para_metrics,
            "delta_gold_chunk": round(
                para_metrics[f"gold_chunk@{top_k}"] - title_metrics[f"gold_chunk@{top_k}"], 4
            ),
            "delta_gold_mrr": round(
                para_metrics["gold_chunk_mrr"] - title_metrics["gold_chunk_mrr"], 4
            ),
        }
        print(
            f"[retrieval:{mode}] title gold_chunk@{top_k}="
            f"{title_metrics[f'gold_chunk@{top_k}']:.3f} -> paraphrase "
            f"{para_metrics[f'gold_chunk@{top_k}']:.3f} "
            f"({report['retrieval'][mode]['delta_gold_chunk']:+.3f})"
        )
    return report


def write_report(preset: RepoPreset, report: dict[str, Any], top_k: int) -> pathlib.Path:
    rows = []
    for mode, metrics in report["retrieval"].items():
        title = metrics["title"]
        para = metrics["paraphrase"]
        rows.append(
            f"| `{mode}` | {title[f'gold_chunk@{top_k}']:.3f} | "
            f"{para[f'gold_chunk@{top_k}']:.3f} | {metrics['delta_gold_chunk']:+.3f} | "
            f"{title['gold_chunk_mrr']:.3f} | {para['gold_chunk_mrr']:.3f} | "
            f"{metrics['delta_gold_mrr']:+.3f} |"
        )
    best_title = max(report["retrieval"], key=lambda m: report["retrieval"][m]["title"][f"gold_chunk@{top_k}"])
    best_para = max(
        report["retrieval"], key=lambda m: report["retrieval"][m]["paraphrase"][f"gold_chunk@{top_k}"]
    )
    body = f"""# Paraphrased queries — does the title advantage hold?

- evaluation items: **{report['n_items']}**
- corpus: `data/processed/{preset.slug}/corpus.jsonl`
- paraphrases: `data/processed/{preset.slug}/paraphrases.jsonl` (one rewrite per item)
- relevance: the maintainer comment that answers the question (gold chunk)

| retriever | gold chunk@{top_k} title | gold chunk@{top_k} paraphrase | Δ | MRR title | MRR paraphrase | Δ |
|---|---|---|---|---|---|---|
{chr(10).join(rows)}

Best retriever by gold chunk@{top_k}: **{best_title}** on titles, **{best_para}** on paraphrases.

## How to read this

Titles are close to the wording the maintainer used, so a lexical retriever gets an
easier case; that was an admitted bias with no measurement behind it until this run.
A negative Δ means the paraphrase removed some of that advantage.

The number to watch is the **relative** drop: if `bm25` falls further than `dense`,
the title advantage was real and the BM25 rows in the main table are optimistic.
"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{preset.slug}__paraphrase.md"
    path.write_text(body, encoding="utf-8")
    json_path = RESULTS_DIR / f"{preset.slug}__paraphrase.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Paraphrase the eval queries and re-measure retrieval")
    parser.add_argument("--preset", default="vite")
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--model", default="")
    parser.add_argument("--limit", type=int, default=0, help="0 = all items")
    parser.add_argument("--measure", action="store_true", help="skip generation, reuse cached paraphrases")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dense-model", default="data/models/bge-base-en-v1.5")
    parser.add_argument("--reranker-model", default="data/models/bge-reranker-base")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    loaded = load_env_file()
    if loaded:
        print(f"loaded from .env: {', '.join(loaded)}")
    preset = load_preset(args.preset)

    if not args.measure:
        generate(preset, provider=args.provider, model_name=args.model, limit=args.limit)
    report = measure(
        preset,
        top_k=args.top_k,
        dense_model=args.dense_model,
        reranker_model=args.reranker_model,
        device=args.device,
    )
    path = write_report(preset, report, args.top_k)
    print(f"\n-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
