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

| | closed-book | with retrieval (BM25) |
|---|---|---|
| scored `correct` | 3.4% | **31.0%** |
| weighted accuracy | 0.138 | **0.483** |
| flagged `fabricated` | 13.8% | 31.0% |

Retrieval roughly triples accuracy. It also *raises* the fabrication rate, which
is the least obvious result in this repo and is discussed under
[Results](#results).

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
Setup: 29 eval items, generator and judge `deepseek-chat`, 5 retrieved excerpts,
`vitejs/vite` corpus of 3,078 documents built from 533 issues and 1,097 comments.

### Answer quality

| system | accuracy | correct | partial | incorrect | fabricated | cited | p50 |
|---|---|---|---|---|---|---|---|
| `closed_book` (no retrieval) | 0.138 | 3.4% | 20.7% | 75.9% | 13.8% | 0% | 4.3 s |
| `bm25` | **0.483** | 31.0% | 34.5% | 34.5% | 31.0% | 100% | 3.8 s |

_accuracy = correct(1.0) + partial(0.5); fabricated = the judge flagged an
invented version, API, or file path._

**Retrieval roughly triples accuracy** (0.14 → 0.48) and removes three quarters of
the unanswerable guesses. That is the headline, and it is the number to quote.

### Where retrieval actually matters (n=28)

Restricted to the items the closed-book model got wrong, i.e. the items where
retrieval has something to add:

| system | accuracy | correct | partial | incorrect | fabricated |
|---|---|---|---|---|---|
| `closed_book` | 0.107 | 0.0% | 21.4% | 78.6% | 14.3% |
| `bm25` | **0.464** | 28.6% | 35.7% | 35.7% | 32.1% |

This is a subgroup of the same run, not a filtered eval set — the full table
above still counts every item, so the comparison is not rigged by construction.

### Retrieval quality — and why this column proves nothing

| retriever | recall@5 | recall@5 leave-one-out | MRR | MRR LOO |
|---|---|---|---|---|
| `bm25` | 1.000 | 0.000 | 1.000 | 0.000 |

**Read this as a broken measurement, not a good one.** The query is the issue
title and every document is prefixed with that same title, so `recall@5 = 1.0` is
close to guaranteed by construction. The leave-one-out column removes the source
issue's own documents and immediately collapses to zero — which is the honest
signal: for these questions the answer exists in exactly one thread, so there is
nothing else to retrieve.

The useful conclusion is a design one, not a numeric one: **this eval set cannot
distinguish retrievers**, because it has no redundancy. A retrieval benchmark
needs questions whose answers appear in more than one thread. What the eval set
*can* measure is end-to-end answer quality, which is why that table is the
headline.

<!-- RESULTS:END -->

### Three things worth noticing in those numbers

**1. Retrieval makes the model more confidently wrong in a new way.**
Fabrication rises from 13.8% to 31.0%. Without retrieval the model usually says
it does not know (75.9% incorrect, but mostly by admitting ignorance). With
retrieval it has plausible specifics in context and misattributes them — a
version number from a neighbouring excerpt restated as the answer. The
conclusion is not "retrieval is bad"; it is that grounding and citation accuracy
are separate problems, and the second one needs its own fix.

**2. A quarter of retrieval answers answer nothing.**
9 of 29 BM25 answers largely describe the thread ("this is tracked in issue
#21827") instead of resolving the question. The prompt was tightened mid-session
to forbid exactly this and it only partly helped. A production system needs an
answer-vs-summary classifier, or a prompt-engineering pass with this as an
explicit metric.

**3. One call stalled for 17 minutes.**
A single generation hung on the network and took 1,031 s. It did not affect the
verdict, but it distorts percentiles at n=29 (p95 hides it entirely). Timeouts
and retries around the generator are missing and should exist before any of this
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

* **LLM-as-judge.** The judge is the same family of model being evaluated. Human
  spot checks over ~30 items are part of the workflow and the agreement rate is
  reported alongside the numbers. Judge scores are a proxy, not ground truth.
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
