# issue-rag — a support assistant that measures itself

A retrieval-augmented question answering system over a real project's issue
tracker, built around one question:

> **How much of what an LLM confidently says about a fast-moving library is
> actually wrong — and does retrieving recent issues fix it?**

Most RAG demos answer the first half by assertion and skip the second. This repo
ships the measurement: a hand-reviewed eval set mined from maintainer answers,
five system configurations, and a results table where every row is an ablation.

---

## The hypothesis

Support traffic on fast-moving libraries is dominated by questions whose answers
live in the **last six to twelve months of issues** — which versions fixed a bug,
which API replaced a deprecated call, why the example in the blog post no longer
compiles.

A model trained before those issues were written cannot know any of it. So it
does what language models do: it produces a plausible version number, a
plausible API name, and a confident tone.

That failure mode is measurable, and it is the thing this project measures.
Measured on 29 hand-reviewed items against `vitejs/vite`:

| | closed-book | with retrieval (best config) |
|---|---|---|
| scored `correct` | 10.3% | **37.9%** |
| weighted accuracy | 0.190 | **0.517** |
| flagged `fabricated` | 3.4% | 34.5% |

Retrieval clearly helps — **but the size of the gain depends on who grades it**, so
this repo reports an interval rather than one number:

- under the in-project judge: **0.196 → 0.518** (2.6×)
- under an independent stricter grader: **0.286 → 0.554** (1.9×)
- under a blind **human** review of 30 answers: **0.100 → 0.567**

The human review is the interesting one: it agrees with the judge on 56.7% of
verdicts (κ=0.607) but disagrees in a consistent direction — the judge gives vague
advice more credit than a person does, and good retrieval answers less. So the
machine-measured gains here are **conservative**, which is the opposite of the usual
worry about LLM judges.

Two things are deliberately *not* claimed. The four retrieval configurations are
mutually indistinguishable (0.466–0.517) — including the reranked one, whose lead
under the original rubric disappeared once the rubric was corrected. And retrieval
*raises* the fabrication rate, which is the least obvious result here. All of it is
discussed under [Results](#results).

---

## Why this repo is not another "chat with your PDF"

Three decisions do the differentiating work, and all three are checkable in the
code:

1. **The eval set is mined from ground truth, not invented.**
   Each question is a real issue title; each reference answer is the maintainer
   reply that closed it. No annotation budget, no LLM-generated questions (which
   would just measure the generating model's taste).

2. **Retrieval is scored with the source issue excluded.**
   If the query can match its own issue thread, recall looks excellent and
   proves nothing. Every retrieval metric is reported twice — once as the
   product number, once leave-one-issue-out as the generalisation number.

3. **A confident wrong answer is scored `incorrect`, never `partial`.**
   The judge rubric ranks fabricated version numbers and API names as the worst
   outcome, because that is precisely the failure this system exists to prevent.

---

## Choosing a target repository (do this first, it can sink the project)

The pipeline is repo-agnostic — `configs/presets.yaml` is the only thing that
knows which project is being studied. But the choice is not cosmetic. Measured on
the 50 most recent comments of each candidate:

| repository | maintainer replies | bot traffic | usable? |
|---|---|---|---|
| `vitejs/vite` | 40% | 20% | **yes** (default preset) |
| `ollama/ollama` | 32% | 0% | yes |
| `vercel/ai` | 8% | 84% | no |
| `langchain-ai/langchainjs` | 2% | 72% | no |

On `vercel/ai`, two bot accounts produced 86% of a 2000-comment sample and 600
issues yielded **10** eval candidates. Two more sampling traps — a window-axis
mismatch that silently emptied the join, and pull-request comments masquerading
as issue comments — are written up with numbers in
[`docs/DATA_QUALITY.md`](docs/DATA_QUALITY.md). Read that file before pointing
this at your own repository.

---

## Architecture

```
GitHub REST API
   ├── /search/issues .................. issue titles + bodies   (100/page)
   └── /repos/{o}/{r}/issues/comments .. maintainer answers      (100/page)
                    │
                    ▼
        corpus.py    issue body chunks + comment chunks,
                     each prefixed with its issue title
                    │
                    ▼
        evalset.py   question = issue title
                     reference = maintainer's accepted reply
                     → human keep/drop → eval set (committed)
                    │
                    ▼
     ┌──────────────┴──────────────┐
     ▼                             ▼
 retrieval.py                  generate.py
 BM25 · dense · RRF hybrid     local Qwen2.5 / any
 · cross-encoder rerank        OpenAI-compatible endpoint
     └──────────────┬──────────────┘
                    ▼
              judge.py    rubric grading against the reference,
                          with a separate grounding audit
                    ▼
             evaluate.py   results.md + results.json
```

Two harvesting details worth knowing, both forced by GitHub's 60-request/hour
anonymous quota:

* issues come from the **search** endpoint, which has its own quota and returns
  bodies;
* comments are fetched **per issue** in `--mode per-issue`. The repo-wide comment
  list is 100× cheaper per comment, but it also returns comments on pull
  requests — on Vite that was ~90% of the stream, and 150 harvested comments
  yielded 16 usable documents. Cheap at producing nothing is still nothing.

There is also a `--mode repo-wide` for repositories where PR traffic is low;
`docs/DATA_QUALITY.md` has the measurement that tells you which applies.

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[models,dev]"

# 1. harvest (works anonymously; set GITHUB_TOKEN for 5000 req/hour instead of 60)
python -m issue_rag.fetch issues   --preset vite --months 12 --allow-wait
python -m issue_rag.fetch comments --preset vite --mode per-issue --max-comment-issues 120

# 2. build the retrieval corpus
python -m issue_rag.corpus --preset vite

# 3. mine question/answer candidates, then review them by hand
python -m issue_rag.evalset --preset vite
#    open data/processed/<repo>/evalset_review.csv, put 1 or 0 in keep_1_or_0
python -m issue_rag.evalset --preset vite --finalize

# 4. download models (ModelScope, because huggingface.co may be unreachable)
python -m issue_rag.models

#    ...or skip the download entirely and use an API for generation/judging:
#    echo "DEEPSEEK_API_KEY=sk-..." >> .env

# 5. run the matrix
python -m issue_rag.evaluate --preset vite --systems closed_book,bm25,dense,hybrid,hybrid_rerank
#    with an API instead of local weights:
#    python -m issue_rag.evaluate --preset vite --provider deepseek
```

`--provider` accepts `local`, `deepseek`, `openai`, `siliconflow`, or `ollama`,
and any other OpenAI-compatible server works via `--endpoint <url>`. Credentials
are read from `.env` (see `.env.example`).

The rerun of any stage is free: HTTP responses and embeddings are cached on disk.

---

## Results

_Produced by `python -m issue_rag.evaluate`. Regenerate with the command above;
numbers are not hand-edited._

<!-- RESULTS:BEGIN -->
Setup: 29 hand-reviewed items, generator and judge `deepseek-chat`, 5 retrieved
excerpts, `vitejs/vite` corpus of 3,078 documents built from 533 issues and 1,097
comments. Rubric ③ (see [The rubric itself was an experiment](#the-rubric-itself-was-an-experiment)).
Full machine-readable output in `data/results/`.

### Answer quality

| system | accuracy | correct | partial | incorrect | fabricated | cited | p50 |
|---|---|---|---|---|---|---|---|
| `closed_book` (no retrieval) | 0.190 | 10.3% | 17.2% | 72.4% | 3.4% | 0% | 4.3 s |
| `bm25` | **0.517** | **37.9%** | 27.6% | 34.5% | 34.5% | 100% | 3.8 s |
| `dense` (bge-base-en-v1.5) | 0.466 | 27.6% | 37.9% | 34.5% | 24.1% | 100% | 3.7 s |
| `hybrid` (RRF) | 0.500 | 37.9% | 24.1% | 37.9% | 37.9% | 100% | 3.7 s |
| `hybrid + rerank` (bge-reranker-base) | 0.500 | **44.8%** | 10.3% | 44.8% | 34.5% | 100% | 3.9 s |

_accuracy = correct(1.0) + partial(0.5); fabricated = the judge flagged a wrong
version, API, or file path._

**Retrieval lifts accuracy from 0.190 to about 0.50, and the four retrieval
configurations are indistinguishable from each other** (0.466–0.517, a spread
smaller than the grader's own noise). That second half matters as much as the
first: this README previously claimed the reranked system was best, and under the
shipped rubric that claim does not survive — see
[What survives](#what-survives-and-what-does-not).

### How much of this is the judge?

Three separate questions, answered by three separate comparisons. All runs reuse
the *same stored answers* (`evaluate --regrade`), so only the grader varies.

_Every figure in this section is **paired**: it uses only the items that have a
usable verdict in both arms, so it can differ from the full tables above by a
hundredth or two. The gain line therefore reads +0.310 here and +0.327 above; both
are computed from the same shards, over a slightly different item set._

**a) Does the rubric matter?** Same model, same answers, rubric ① vs ③:

| system | rubric ① | rubric ③ | delta |
|---|---|---|---|
| `closed_book` | 0.138 | 0.190 | +0.052 |
| `bm25` | 0.483 | 0.517 | +0.035 |
| `dense` | 0.448 | 0.466 | +0.017 |
| `hybrid` | 0.466 | 0.500 | +0.035 |
| `hybrid + rerank` | 0.586 | 0.500 | **−0.086** |
| **best retrieval − closed_book** | **+0.448** | **+0.310** | |

Almost every system moves by a few hundredths; the reranked one moves by −0.086,
which is the whole reason its apparent lead disappears.

**b) Does the judge identity matter?** Same rubric ③, `deepseek-chat` vs
`deepseek-reasoner`, on the two systems measured:

| system | `deepseek-chat` | `deepseek-reasoner` | delta |
|---|---|---|---|
| `closed_book` | 0.196 | 0.286 | +0.089 |
| `hybrid + rerank` | 0.518 | 0.554 | +0.036 |
| **gain** | **+0.322** (2.6×) | **+0.268** (1.9×) | |

**c) Does the judge agree with itself?** Two runs of `deepseek-reasoner` over
identical answers, rubric ③:

| system | run a | run b | delta | exact agreement |
|---|---|---|---|---|
| `closed_book` | 0.287 | 0.292 | 0.005 | 70.4% |
| `hybrid + rerank` | 0.567 | 0.558 | **0.010** | 82.1% |

Compare rubric ①, where the same measurement gave 0.069 and 0.017 on those two
systems. Tightening the rubric cut the run-to-run difference on the reported
aggregate by roughly 2–7× **and** cut the cross-judge gain interval from
1.5×–4.2× to **1.9×–2.6×**.

**d) Does a *human* agree?** The only measurement here that is not a model grading
a model. 30 answers (15 per system) were graded blind through a local review page:
the reviewer saw the question, the reference and the candidate, but never the
judge's verdict and never which system produced the answer.

| metric | value |
|---|---|
| reviewed pairs | 30 |
| exact agreement | **56.7%** |
| usable-vs-not agreement | **66.7%** |
| quadratic-weighted kappa | **0.607** ("substantial") |

| system | n | exact | usable | mean human | mean judge |
|---|---|---|---|---|---|
| `closed_book` | 15 | 47% | 53% | **0.100** | 0.267 |
| `hybrid + rerank` | 15 | 67% | 80% | **0.567** | 0.467 |

The last two columns are the finding. **The judge is lenient on vague advice and
strict on good answers**: it scores closed-book 0.167 *higher* than the human does
and the reranked system 0.100 *lower*. So on this sample the human sees a **+0.467**
gap where the judge sees +0.200 — the machine-measured gain is a *conservative*
estimate, not an inflated one. That is the opposite direction from the usual worry
about LLM judges.

Caveats, because one sample is one sample: a single reviewer, 15 per system, one
sitting. Treat it as validating the instrument's *direction*, not as a precise
estimate.

### What survives, and what does not

| claim | verdict |
|---|---|
| Retrieval beats closed-book | **holds** — every rubric, every judge, every run, and the human review |
| The gain is "4×" | **does not hold** — it is **1.9×–2.6×** by the model judges; the human sample suggests larger |
| The reranked system is the best retrieval config | **does not hold** — it leads under rubric ① and ties or trails under ③ |
| Any ordering among `bm25` / `hybrid` / `dense` / `rerank` | **unresolved** — all within 0.05, i.e. inside the noise floor |
| The judge understates the gain | **supported** — it over-credits vague advice by 0.167 and under-credits retrieval by 0.100 |

The ordering row is well supported rather than merely suspected: the four retrieval
configurations sit within 0.05 of each other, the grader's run-to-run noise is
~0.01–0.07, and the retrieval metrics are degenerate (everything finds the gold
document at rank 1). Three independent lines of evidence say the same thing.

### On the items the closed-book model got wrong

| system | accuracy | correct | incorrect | fabricated |
|---|---|---|---|---|
| `closed_book` | 0.096 | 0.0% | 80.8% | 3.8% |
| `bm25` | 0.481 | 34.6% | 38.5% | 34.6% |
| `dense` | 0.442 | 26.9% | 38.5% | 26.9% |
| `hybrid` | 0.519 | 38.5% | 34.6% | 34.6% |
| `hybrid + rerank` | 0.500 | 46.2% | 46.2% | 34.6% |

A subgroup of the same run (n=26), not a filtered eval set — the table above still
counts every item, so the comparison is not rigged by construction.

### Retrieval quality — and why these numbers cannot rank retrievers

| retriever | recall@5 | recall@5 leave-one-out | MRR | MRR LOO |
|---|---|---|---|---|
| `bm25` | 1.000 | 0.000 | 1.000 | 0.000 |
| `dense` | 1.000 | 0.000 | 1.000 | 0.000 |
| `hybrid` | 1.000 | 0.000 | 1.000 | 0.000 |
| `hybrid_rerank` | 1.000 | 0.000 | 1.000 | 0.000 |

**Read this as a broken measurement, not a good one.** The query is the issue
title and every document is prefixed with that same title, so the gold document
is found at rank 1 by *every* retriever. The leave-one-out column removes the
source issue's own documents and collapses to zero, which is the honest signal:
each answer exists in exactly one thread, so there is no second thread to find.

<!-- RESULTS:END -->

### The rubric itself was an experiment

The judge above asked for a verdict directly. Two runs of the same model disagreed
on 7–29% of items, and the disagreements clustered on the partial/incorrect
boundary — precisely the decision a holistic prompt leaves undefined. So the
grader was rewritten to answer four factual sub-questions, with the verdict mapped
in code (`judge.verdict_from_checks`).

The first attempt made things worse in an instructive way. Each arm below is two
runs of `deepseek-reasoner` over the same stored answers:

| rubric | closed_book | hybrid+rerank | run-to-run Δ (rerank) | exact agreement (rerank) |
|---|---|---|---|---|
| ① holistic (original) | 0.336 | 0.517 | 0.069 | 82.8% |
| ② four checks, `wrong_specific` = "any detail absent from the reference" | 0.034 | 0.259 | 0.069 | 93.1% |
| ③ four checks, `wrong_specific` = "a version/issue number the reference contradicts" | 0.287 | **0.562** | **0.018** | 82.1% |

**Arm ② is the cautionary one.** It raised exact agreement to 93% and destroyed
both scores, because "absent from the reference" is not "invented" — a retrieval
answer legitimately quotes details from other excerpts, and the broad check flagged
97% of closed-book answers and 93% of retrieval answers as fabricating. Reproducible
and wrong is worse than noisy and roughly right.

**Arm ③** narrows the check to version and issue numbers only, which drops the
flag rate to 0% / 19%. The scores return and the run-to-run difference on the
system that matters falls 3.8× (0.069 → 0.018).

One honest limit: arm ③ does **not** improve per-item agreement (82.1% vs 82.8%).
What it improves is the *net* shift between runs — the errors cancel instead of
biasing the aggregate. Since every number reported here is an aggregate, that is
the variance that mattered.

> **Which rubric produced the tables above?** All of them: the whole run was
> re-baselined from rubric ① into ③ over the original stored answers
> (`evaluate --regrade --regrade-from v1holistic`), so the numbers on this page and
> the code in the repo agree. The ① arm is archived as `v1holistic` and is what the
> "rubric ① vs ③" comparison above reads.

### Five things worth noticing in those numbers

**1. The reranker's advantage did not survive a rubric change.** Under rubric ① it
was the best system by 0.10; under ③ it ties with plain hybrid and trails BM25. The
rubric moved it by −0.086 while every other system moved by +0.02 to +0.05. A
result that flips when the measuring instrument is corrected was a result about the
instrument. This is the single most useful thing in the repo to have caught.

**2. The four retrieval configurations are indistinguishable.** 0.466, 0.500, 0.500,
0.517 — a 0.05 spread against a grader whose own run-to-run noise is 0.01–0.07.
Three independent lines point the same way: the spread is tiny, the retrieval
metrics are degenerate (gold always at rank 1), and every result is within one or
two items at n=29. Do not claim an ordering.

**3. Retrieval buys roughly 2×, and that is the durable finding.** closed_book
moves from 0.190 to about 0.50 under the shipped rubric, and the gain survives two
judges (+0.322 vs +0.268) and every rubric tried. The interval is 1.9×–2.6×, not
"4×".

**4. Retrieval makes the model confidently wrong rather than vague.** With
retrieval, fabrication sits at 24–38% and citation markers appear in 100% of
answers; without it, the model mostly admits ignorance and fabricates only 3.4% of
the time. Grounding and citation fidelity are separate problems and the second one
is unsolved here.

**5. About a quarter of retrieval answers summarise instead of answering.**
Nine of 29 BM25 answers were of the form "this is tracked in issue #21827". The
prompt forbids exactly that and only partly helped. In a real product this needs
an answer-vs-summary check, not a bigger model.


---

## Design decisions and their trade-offs

**Retrieval unit = one document per comment, prefixed with the issue title.**
A maintainer reply is usually self-contained ("on v5 call
`convertToModelMessages` first"), so it retrieves and cites well alone. The title
prefix keeps chunk-level references interpretable.

**Query = issue title, not issue body.** Using the body lets the retriever match
the issue's own text and inflates every metric. The title is also closer to what
a real user types.

**RRF instead of score normalisation.** BM25 scores and cosine similarity are not
comparable; percentile or min-max normalisation is sensitive to the candidate
pool size. Reciprocal rank fusion only uses positions, which makes the hybrid
robust and the ablation interpretable.

**A local 3B model as the default generator.** Small enough to run on a 16 GB
laptop GPU, so anyone can reproduce the table without an API key. The trade-off
is real: absolute scores are lower than a frontier model's. `--endpoint` swaps in
any OpenAI-compatible server, and the closed-book vs RAG *gap* is the finding —
that gap survives a stronger generator.

---

## Limitations

Stated plainly, because a results table without limitations is marketing.

* **LLM-as-judge, now with a human check.** Swapping the judge moves a system by up
  to 0.09, the *same* judge run twice moves it by up to 0.07, and changing the
  rubric moved one system by 0.086. A blind human review of 30 pairs agrees with the
  judge on 56.7% of verdicts (quadratic kappa 0.607) and, more usefully, disagrees
  in a consistent direction: the judge over-credits vague advice by 0.167 and
  under-credits retrieval by 0.100. The remaining limit is *coverage* — one
  reviewer, 15 items per system, one sitting — not the absence of a check.
* **Mined eval set, not a curated benchmark.** Maintainer replies are occasionally
  incomplete or answer a narrower question than the title suggests. The
  keep/drop review pass exists to cut those, and the sheet is committed so the
  selection is auditable.
* **One repository, one domain.** Conclusions about BM25-versus-dense weighting
  are domain-specific; a different preset will reorder the table.
* **Title-as-query favours lexical matching.** BM25 gets a slightly easier task
  than it would with paraphrased questions. The leave-one-out column partially
  compensates, but a paraphrase-based query set is the honest next step.
* **No multi-hop.** Every question is answered from a single issue thread. Questions
  that need two issues combined are out of scope.

---

## Repository layout

```
configs/presets.yaml      target repositories (swap the domain here)
src/issue_rag/
  config.py               paths + preset loading
  github.py               caching, quota-aware REST client
  fetch.py                issue + comment harvest
  corpus.py               cleaning, chunking, document construction
  evalset.py              candidate mining + human review loop
  retrieval.py            BM25 / dense / RRF hybrid / rerank
  generate.py             local HF model + OpenAI-compatible backend
  judge.py                rubric grading + grounding audit
  evaluate.py             the experiment matrix and report writer
  models.py               weight download (ModelScope or HF)
  serve.py                dependency-free demo UI
tests/test_pipeline.py    hermetic tests: chunking, tokenizer, RRF, filters, parsing
docs/DATA_QUALITY.md      the sampling traps, measured
docs/RUNBOOK.md           the two-week execution plan, stage by stage
```

---

## License

MIT.
