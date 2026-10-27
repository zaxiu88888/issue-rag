# Runbook — from zero to published numbers

Status as of the last working session is marked inline. Read
[`DATA_QUALITY.md`](DATA_QUALITY.md) first if you have not: three of the stages
below exist purely because of the failures documented there.

---

## Where the project stands

| stage | state |
|---|---|
| harvest client (cache, quota, retry, gzip) | done |
| issue + comment harvest | done for `vite`, needs ~2 more hours of quota for a 100-item eval set |
| corpus build (chunking, bot filter, title prefix) | done |
| eval-set mining + human review sheet | done, waiting on harvest volume |
| retrieval (BM25 / dense / RRF / rerank) | done |
| generation + judge + results report | done, not yet run (needs weights) |
| unit tests | 27 passing |
| CI | configured |

Two things are missing and both are waits, not work: **GitHub quota** and
**model weights**.

---

## 1. Harvest more data (the only real bottleneck)

Anonymous GitHub gives 60 core requests per hour. `--mode per-issue` spends one
per issue, so:

```
60 issues/hour anonymous   →  100 eval items needs ~2 hours
5000 requests/hour with a token  →  minutes
```

A fine-grained token with **public repository read-only** access is enough. Put
it in `GITHUB_TOKEN` and rerun the same command; every already-cached page is
free, so nothing is re-downloaded.

```bash
python -m issue_rag.fetch comments --preset vite --mode per-issue --max-comment-issues 120
```

Re-running is safe and resumable. Stop and restart as often as you like.

Then rebuild and re-mine:

```bash
python -m issue_rag.corpus  --preset vite
python -m issue_rag.evalset --preset vite
```

---

## 2. Review the eval set by hand (do not skip)

```bash
# opens in Excel / Numbers / any CSV editor
data/processed/vitejs__vite/evalset_review.csv
```

Put `1` in `keep_1_or_0` for a question that is:

- answerable from the reference answer alone,
- specific enough that a vague answer is clearly wrong,
- something a real user would plausibly ask.

Put `0` for anything where the maintainer reply is triage boilerplate
("duplicate of #123"), or where the answer is only understandable with a linked
thread you do not have.

**Target 80–120 kept items.** Fewer than 50 and the confidence intervals are too
wide to claim anything; the report prints `n` next to every metric for that
reason.

Then:

```bash
python -m issue_rag.evalset --preset vite --finalize
```

This writes `evalset.jsonl`, which is committed to the repository. That file is
the project's real artifact — the code is replaceable, a clean hand-checked eval
set is not.

---

## 3. Weights

`huggingface.co` was unreachable on the machine this was built on; ModelScope is
the default backend.

```bash
python -m issue_rag.models                  # all three
python -m issue_rag.models --only dense     # embedding model only, fastest path
```

Sizes and what each unlocks:

| model | size | unlocks |
|---|---|---|
| `bge-base-en-v1.5` | ~440 MB | `dense`, `hybrid`, `hybrid_rerank` rows |
| `bge-reranker-base` | ~1.1 GB | the `hybrid_rerank` row |
| `Qwen2.5-3B-Instruct` | ~6.2 GB | the answer-quality table |

You can produce the **entire retrieval table** with only the first model. Do that
first: it is the part of the results that does not depend on the generator.

---

## 4. Run the experiments

Retrieval only — no generator needed, minutes on CPU:

```bash
python -m issue_rag.evaluate --preset vite \
  --systems bm25,dense,hybrid,hybrid_rerank --retrieval-only
```

Full matrix — this is the headline table. On a 16 GB GPU with the 3B model it is
roughly 8–12 s per item per system, so 100 items × 5 systems ≈ 1.5–2 hours. It is
resumable per (system, item): interrupt it freely and rerun.

```bash
python -m issue_rag.evaluate --preset vite \
  --systems closed_book,bm25,dense,hybrid,hybrid_rerank
```

Start with a smoke test so a crash surfaces in minute two instead of hour two:

```bash
python -m issue_rag.evaluate --preset vite --limit 5 --systems closed_book,bm25
```

---

## 5. Human-verify the judge (required before publishing)

The judge is the same model family as the generator. Grade 30 items yourself
against `reference_answer` and compare with the judge's verdicts in the shard
files under `data/results/`. Report the agreement rate in the README. If
agreement is below ~80%, the rubric is too loose — tighten it in
`judge.py:GRADE_SYSTEM` before believing any of the numbers.

---

## 6. Publish

- Paste the generated `data/results/<repo>__results.md` tables into the README
  `RESULTS` block (between the HTML comments).
- Add the limitations you actually hit, not the generic ones.
- Write the technical post: the interesting part is
  [`DATA_QUALITY.md`](DATA_QUALITY.md), not the accuracy number.
- Commit daily. The commit history is evidence the project is real.

---

## Timing expectation

Two weeks at 2–4 h/day, realistic split:

| block | hours | what |
|---|---|---|
| environment + harvest plumbing | 4 | done |
| harvest enough data | 2 (mostly waiting) | network-bound |
| eval-set review | 3 | the highest-value hours in the project |
| retrieval experiments | 3 | cheap, fast iteration |
| generation experiments | 4 | mostly waiting on the GPU |
| README, post, polish | 4 | do not skip this |
| slack | 6 | something will break |
