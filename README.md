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

| | closed-book | with retrieval (hybrid + rerank) |
|---|---|---|
| scored `correct` | 3.4% | **44.8%** |
| weighted accuracy | 0.138 | **0.586** |
| flagged `fabricated` | 13.8% | 31.0% |

Retrieval clearly helps — **but the size of the gain depends on who grades it**,
so this repo reports an interval rather than one number:

- under the in-project judge: **0.138 → 0.586** (4.2×)
- under an independent stricter grader: **0.328 → 0.483** (1.5×)

What *does* hold under both graders is the shape of the result: closed-book is
worst and the reranked system is best. The ordering among the three plain
retrievers is **not** stable and is reported as unresolved, because they sit
within one item of each other and the grader has its own run-to-run noise.
Retrieval also *raises* the fabrication rate, which is the least obvious result
here. All of it is discussed under [Results](#results).

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
comments. Full machine-readable output in `data/results/`.

### Answer quality

| system | accuracy | correct | partial | incorrect | fabricated | cited | p50 |
|---|---|---|---|---|---|---|---|
| `closed_book` (no retrieval) | 0.138 | 3.4% | 20.7% | 75.9% | 13.8% | 0% | 4.3 s |
| `bm25` | 0.483 | 31.0% | 34.5% | 34.5% | 31.0% | 100% | 3.8 s |
| `dense` (bge-base-en-v1.5) | 0.448 | 31.0% | 27.6% | 41.4% | 34.5% | 100% | 3.7 s |
| `hybrid` (RRF) | 0.466 | 31.0% | 31.0% | 37.9% | 27.6% | 100% | 3.7 s |
| `hybrid + rerank` (bge-reranker-base) | **0.586** | **44.8%** | 27.6% | 27.6% | 31.0% | 100% | 3.9 s |

_accuracy = correct(1.0) + partial(0.5); fabricated = the judge flagged an
invented version, API, or file path._

### How much of this is the judge?

The judge above is `deepseek-chat`, the same model that wrote the answers —
circular. So every answer was re-graded by a **different model**,
`deepseek-reasoner`, over the *same stored answers* (`evaluate --regrade`), and
then the reasoner was run a **second time** to measure its own noise. All three
runs are committed under `data/results/` and regenerate with `make regrade` +
`make sensitivity`.

**Run A vs run B (different model, same answers):**

| system | `deepseek-chat` | `deepseek-reasoner` | delta |
|---|---|---|---|
| `closed_book` | 0.138 | 0.328 | **+0.190** |
| `bm25` | 0.483 | 0.414 | −0.069 |
| `dense` | 0.448 | 0.362 | −0.086 |
| `hybrid` | 0.466 | 0.466 | 0.000 |
| `hybrid + rerank` | 0.586 | 0.483 | −0.103 |
| **closed_book → best RAG** | **+0.448** (4.2×) | **+0.155** (1.5×) | |

**Run B vs run C (same model twice — the noise floor):**

| system | `deepseek-reasoner` [a] | `deepseek-reasoner` [b] | delta |
|---|---|---|---|
| `closed_book` | 0.328 | 0.345 | +0.017 |
| `bm25` | 0.414 | 0.397 | −0.017 |
| `dense` | 0.339 | 0.375 | +0.036 |
| `hybrid` | 0.466 | 0.431 | −0.035 |
| `hybrid + rerank` | 0.483 | 0.552 | **+0.069** |

Two runs of the *same* grader agree on only **71–93%** of verdicts depending on
the system. So the grader is itself stochastic, and **~0.07 accuracy is noise**;
differences smaller than that between two systems are not results.

### What survives, and what does not

| claim | verdict |
|---|---|
| Retrieval beats closed-book | **holds** — every run, every grader |
| The reranked system is the best retrieval config | **holds** — in all three runs |
| The gain is "4×" | **does not hold** — it is **1.5×–4.2×** |
| `bm25` vs `hybrid` vs `dense` ordering | **unresolved** — they swap between graders and sit within 0.07 of each other |

The mechanism shows up in the transitions: the stricter grader gives partial
credit to generic advice (lifting closed-book by 0.19) and withdraws it from
answers that pair a correct fix with an invented version number (lowering the
reranked system by 0.10). Neither grader is ground truth, and the three runs
together are the honest estimate rather than any one of them.

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

> **Which rubric produced the tables above?** Rubric ①, kept so the earlier runs
> stay comparable. The repo now ships ③. Re-running `make eval` with it moves the
> two extremes by about −0.05 (closed_book) and +0.05 (reranked) on two measured
> systems, which slightly *increases* the gain; the RUNBOOK makes that re-baseline
> a prerequisite for publishing new numbers.

### On the 28 items the closed-book model got wrong

| system | accuracy | correct | incorrect | fabricated |
|---|---|---|---|---|
| `closed_book` | 0.107 | 0.0% | 78.6% | 14.3% |
| `bm25` | 0.464 | 28.6% | 35.7% | 32.1% |
| `dense` | 0.464 | 32.1% | 39.3% | 35.7% |
| `hybrid` | 0.464 | 32.1% | 39.3% | 28.6% |
| `hybrid + rerank` | **0.571** | 42.9% | 28.6% | 32.1% |

A subgroup of the same run, not a filtered eval set — the table above still counts
every item, so the comparison is not rigged by construction.

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

The consequence is important for reading the table above: **the four retrieval
configurations differ only in what the *other* four excerpts are**, not in
whether the right issue is found. A retrieval benchmark that can rank retrievers
needs questions whose answers appear in several threads, or paraphrased queries.

<!-- RESULTS:END -->

### Five things worth noticing in those numbers

**1. Retrieval helps, but by how much depends on who grades, and the grader is
itself noisy.** Under the in-project judge the gain is 0.138 → 0.586 (4.2×);
under a second, stricter model it is 0.328 → 0.483 (1.5×); and two runs of that
second model disagree with each other on 7–29% of verdicts. What survives all
three runs is the *shape*: closed-book worst, reranked best. What does not is any
ordering among the three plain retrievers, or any single number. See
[How much of this is the judge](#how-much-of-this-is-the-judge).

**2. The retriever choice is noise; the reranker is not.** BM25, dense, and hybrid
land at 0.483 / 0.448 / 0.466 — a spread of about one item at n=29, i.e. nothing.
Given that every retriever finds the gold issue at rank 1, that is exactly what
you would expect. Reranking moves hybrid from 0.466 to 0.586 (+0.12, about three
items), which is the largest gap in the retrieval half and worth **one** more
dataset before believing it. The plausible mechanism is real, though: the
first-stage retriever already picks the right thread, so the only lever left is
which *supporting* excerpts make it into the context, and that is precisely what
a cross-encoder orders.

**3. Retrieval makes the model more confidently wrong.** Fabrication rises from
13.8% to 27–34%. Without retrieval the model usually admits ignorance ("I don't
have reliable knowledge of a Vite 8 release") — wrong, but not invented. With
retrieval it has plausible specifics in context and misattributes them. Grounding
and citation fidelity are separate problems, and the second one is unsolved here.

**4. About a quarter of retrieval answers summarise instead of answering.**
Nine of 29 BM25 answers were of the form "this is tracked in issue #21827". The
prompt forbids exactly that and only partly helped. In a real product this needs
an answer-vs-summary check, not a bigger model.

**5. One generation call stalled for 17 minutes.** A single request hung and took
1,031 s. It did not change a verdict, but at n=29 it is invisible in p95 and it
means the generator has no timeout or retry. That has to exist before any of this
runs unattended.


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

* **LLM-as-judge, and the grader is noisy as well as debatable.** Two different
  models agree on only 48%–93% of verdicts, and — the part that is easy to miss —
  the *same* model run twice agrees with itself on only 71%–93%. That self-noise
  is worth ~0.07 accuracy, so system differences below that threshold are not
  interpretable here. Three runs are committed (`data/results/*sensitivity*`), the
  gain is reported as a 1.5×–4.2× interval, and the claims that survive are
  separated from the ones that do not. What is still *missing* is a **human**
  validation: no human has graded these items, so "agreement with a second model"
  must not be read as "agreement with the truth".
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
