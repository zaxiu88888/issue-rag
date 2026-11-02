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
Measured on **95 hand-reviewed items** against `vitejs/vite`:

| | closed-book | with retrieval (best config) |
|---|---|---|
| scored `correct` | 8.4% | **64.2%** |
| weighted accuracy | 0.221 | **0.700** |
| flagged `fabricated` | 1.1% | 20.0% |

Retrieval clearly helps — **but the size of the gain depends on who grades it**, so
this repo reports an interval rather than one number:

- under the in-project judge (`deepseek-chat`): **0.221 → 0.700** (3.2×)
- under an independent stricter grader (`deepseek-reasoner`): **0.305 → 0.711** (2.3×)
- under a blind **human** review of 30 answers: **0.167 → 0.867** (5.2×)

The human review is the interesting one: it agrees with the judge on **70.0%** of
verdicts (quadratic κ=0.816) but disagrees in a consistent direction — the judge
gives vague advice more credit than a person does (+0.100) and good retrieval
answers less (−0.133). So the machine-measured gains here are **conservative**,
which is the opposite of the usual worry about LLM judges.

Expanding the eval set is itself an experiment, and it is reported as one. Going
from 29 to 95 items moved closed-book by **+0.031** — while every retrieval system
moved by +0.17 to +0.20 — so the headline effect got *stronger* with more data
rather than regressing toward zero. See [29 items vs 95](#29-items-vs-95-items-what-expanding-the-sample-changed).

Two things are deliberately *not* claimed. The four retrieval configurations are
mutually indistinguishable (0.632–0.700, a spread smaller than the grader's own
noise, and the ordering flips when the grader changes). And retrieval *raises* the
fabrication rate, which is the least obvious result here. All of it is discussed
under [Results](#results).

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
Setup: **95 hand-reviewed items**, generator and judge `deepseek-chat`, 5 retrieved
excerpts, `vitejs/vite` corpus of **5,473 documents** built from **771 issues and
2,593 comments**. Rubric ③ (see
[The rubric itself was an experiment](#the-rubric-itself-was-an-experiment)).
Full machine-readable output in `data/results/`.

### Answer quality

| system | accuracy | correct | partial | incorrect | fabricated | cited | p50 | p95 |
|---|---|---|---|---|---|---|---|---|
| `closed_book` (no retrieval) | 0.221 | 8.4% | 27.4% | 64.2% | 1.1% | 0% | 4.32 s | 5.00 s |
| `bm25` | **0.700** | **64.2%** | 11.6% | 24.2% | 20.0% | 100% | 3.70 s | 4.29 s |
| `dense` (bge-base-en-v1.5) | 0.632 | 55.8% | 14.7% | 29.5% | 28.4% | 100% | 3.65 s | 4.29 s |
| `hybrid` (RRF) | 0.695 | 63.2% | 12.6% | 24.2% | 21.1% | 100% | 3.53 s | 4.19 s |
| `hybrid + rerank` (bge-reranker-base) | 0.679 | 62.1% | 11.6% | 26.3% | 23.2% | 100% | 3.77 s | 4.33 s |

_accuracy = correct(1.0) + partial(0.5); fabricated = the judge flagged a wrong
version, API, or file path. Every row is n=95 with 0 failed calls._

**Retrieval lifts accuracy from 0.221 to about 0.70, and the four retrieval
configurations are indistinguishable from each other** (0.632–0.700, a spread
smaller than the grader's own noise). That second half matters as much as the
first: this README once claimed the reranked system was best, and under the
shipped rubric that claim does not survive — see
[What survives](#what-survives-and-what-does-not).

### 29 items vs 95 items: what expanding the sample changed

The previous revision of this README reported 29 hand-reviewed items. The set was
then rebuilt from a wider window (18 months of issues, 3,078 → 5,473 documents)
and re-reviewed item by item: 200 candidates, 95 kept, 105 dropped. Same rubric,
same generators, same command. What changed:

| system | 29 items | 95 items | Δ |
|---|---|---|---|
| `closed_book` | 0.190 | 0.221 | **+0.031** |
| `bm25` | 0.517 | 0.700 | +0.183 |
| `dense` | 0.466 | 0.632 | +0.166 |
| `hybrid` | 0.500 | 0.695 | +0.195 |
| `hybrid + rerank` | 0.500 | 0.679 | +0.179 |
| **gain (best retrieval ÷ closed-book)** | **2.72×** | **3.17×** | +0.45× |

Two readings, and both are honest:

1. **The headline is stable, and it got stronger, not weaker.** Closed-book moved
   by three points on a sample 3.3× larger; retrieval moved by ~0.18. A finding
   that survived a 3× increase in sample size and came out *larger* is not a
   small-sample artefact of the kind reviewers rightly suspect.
2. **The absolute numbers are not comparable across revisions.** The wider window
   added many more recent, well-answered threads, and retrieval benefits from that
   far more than the closed-book model does. Do not compare a row on this page to a
   row in an older screenshot.

The confidence interval also narrowed by construction: at n=29 a single item moves
a system by 3.4 points; at n=95 it moves it by 1.1.

### How much of this is the judge?

Three separate questions, answered by three separate comparisons. All runs reuse
the *same stored answers* (`evaluate --regrade`), so only the grader varies.

**a) Does the rubric matter?** Same model, same answers, rubric ① vs ③ — measured
on the *previous* 29-item set, because re-implementing rubric ① is a change to the
instrument and is not mixed into the current run:

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

**b) Does the judge identity matter?** Same rubric ③, same 95 stored answers,
`deepseek-chat` vs `deepseek-reasoner`:

| system | `deepseek-chat` | `deepseek-reasoner` | delta |
|---|---|---|---|
| `closed_book` | 0.221 | 0.305 | **+0.084** |
| `bm25` | 0.700 | 0.658 | −0.042 |
| `dense` | 0.632 | 0.595 | −0.037 |
| `hybrid` | 0.695 | 0.668 | −0.026 |
| `hybrid + rerank` | 0.679 | **0.711** | **+0.032** |
| **gain (best retrieval ÷ closed-book)** | **3.17×** | **2.33×** | |

The stricter grader is *lenient* on closed-book answers and *strict* on retrieval
answers, which compresses the gain. It also reorders the table: `bm25` is best
under `deepseek-chat`, `hybrid + rerank` is best under `deepseek-reasoner`, and the
largest mover between them is 0.042 — inside the noise band measured next.

**c) Does the judge agree with itself?** Two runs of `deepseek-reasoner` over the
same 95 answers (this is the noise floor: a difference between two systems smaller
than this is not a result):

| system | run a | run b | delta | exact agreement |
|---|---|---|---|---|
| `closed_book` | 0.305 | 0.289 | −0.016 | 78.9% |
| `bm25` | 0.658 | 0.668 | +0.011 | 86.3% |
| `dense` | 0.595 | 0.621 | +0.026 | 84.2% |
| `hybrid` | 0.668 | 0.684 | +0.016 | 86.3% |
| `hybrid + rerank` | 0.711 | 0.705 | −0.005 | 86.3% |

Run-to-run drift is **0.005–0.026** on the aggregate, and the ordering is *identical*
in both runs. Cross-judge agreement is lower (53.7%–83.2% exact) — two different
models disagree more than one model disagrees with itself, which is why the interval
above is reported as an interval.

**d) Does a *human* agree?** The only measurement here that is not a model grading
a model. 30 answers (15 per system) were graded blind through a local review page:
the reviewer saw the question, the reference and the candidate, but never the
judge's verdict and never which system produced the answer.

| metric | value |
|---|---|
| reviewed pairs | 30 |
| exact agreement | **70.0%** |
| usable-vs-not agreement | **73.3%** |
| quadratic-weighted kappa | **0.816** ("almost perfect") |

| system | n | exact | usable | mean human | mean judge |
|---|---|---|---|---|---|
| `closed_book` | 15 | 67% | 73% | **0.167** | 0.267 |
| `hybrid + rerank` | 15 | 73% | 73% | **0.867** | 0.733 |

The last two columns are the finding. **The judge is lenient on vague advice and
strict on good answers**: it scores closed-book answers 0.100 *higher* than the
human does and the reranked system 0.133 *lower*. So on this sample the human sees
a **+0.700** gap (5.2×) where the judge sees +0.466 (2.75×) — the machine-measured
gain is a *conservative* estimate, not an inflated one. That is the opposite
direction from the usual worry about LLM judges.

Caveats, because one sample is one sample: a single reviewer, 15 per system, one
sitting. Treat it as validating the instrument's *direction*, not as a precise
estimate. Worth noting that both this axis and the sample-size experiment point the
same way — the measured retrieval gain is a floor.

### What survives, and what does not

| claim | verdict |
|---|---|
| Retrieval beats closed-book | **holds** — every rubric, every judge, every run, every sample size, and the human review |
| The gain is "4×" | **does not hold** — it is **2.3×–3.2×** by the model judges; the human sample suggests ~5× |
| The reranked system is the best retrieval config | **does not hold** — it leads under `deepseek-reasoner`, trails `bm25` under `deepseek-chat`, and the gap is 0.03 |
| Any ordering among `bm25` / `hybrid` / `dense` / `rerank` | **unresolved** — all within 0.07, i.e. at or inside the noise floor |
| The judge understates the gain | **supported** — it over-credits vague advice by 0.100 and under-credits retrieval by 0.133 |
| The eval set was too small to trust | **does not hold** — tripling it moved closed-book by 0.031 |

The ordering row is well supported rather than merely suspected: the four retrieval
configurations sit within 0.07 of each other, the grader's run-to-run noise is
0.005–0.026 on the aggregate, the ordering flips when the grader changes, and the
issue-level retrieval metrics are degenerate (see below). Four independent lines of
evidence say the same thing: do not claim an ordering.

### On the items the closed-book model got wrong

| system | accuracy | correct | partial | incorrect | fabricated |
|---|---|---|---|---|---|
| `closed_book` | 0.149 | 0.0% | 29.9% | 70.1% | 1.1% |
| `bm25` | 0.678 | 62.1% | 11.5% | 26.4% | 21.8% |
| `dense` | 0.621 | 54.0% | 16.1% | 29.9% | 28.7% |
| `hybrid` | 0.672 | 60.9% | 12.6% | 26.4% | 21.8% |
| `hybrid + rerank` | 0.672 | 60.9% | 12.6% | 26.4% | 24.1% |

A subgroup of the same run (n=87), not a filtered eval set — the table above still
counts every item, so the comparison is not rigged by construction. This is where
retrieval actually does its work: on items the model can already answer from memory
it adds nothing, and 8 of 95 items are in that category.

### Retrieval quality — measured on the answering comment, not the issue

Relevance here is the **comment that answers the question**, not the issue thread it
came from:

| retriever | gold chunk@5 | gold chunk MRR | gold chunk top-1 | share of window from the same issue |
|---|---|---|---|---|
| `bm25` | 0.611 | 0.280 | 0.116 | 0.983 |
| `dense` | 0.516 | 0.206 | 0.063 | 0.998 |
| `hybrid` | 0.579 | 0.247 | 0.084 | 0.996 |
| `hybrid + rerank` | **0.621** | **0.289** | **0.147** | 0.990 |

This table does discriminate — it separates `dense` (0.516) from the reranker
(0.621), which the answer-quality table cannot do. It also produces the most
interesting negative result in the repo: **the retriever with the best retrieval
quality is not the one with the best answers** (`hybrid + rerank` leads here at
0.621 but trails `bm25` on accuracy, 0.679 vs 0.700). Better retrieval of the
reference comment does not fully propagate to the final answer, which is the
honest reason the four configurations cannot be ranked.

The last column explains the older, issue-level table: **98–99.8% of the retrieved
window is chunks of the gold issue itself.** The retriever finds the right thread
(trivially — the query is the thread's title) and then fills all five slots with
that thread. Any issue-level "was the right issue retrieved" metric is therefore
~1.0 by construction for every retriever, which is why it was replaced.

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

**2. The four retrieval configurations are indistinguishable.** 0.632, 0.679, 0.695,
0.700 — a 0.07 spread against a grader whose own run-to-run noise is 0.005–0.026
on the aggregate, with the ordering flipping when the judge changes. Four
independent lines point the same way: the spread is small, the ordering is not
stable across judges, the issue-level retrieval metrics are degenerate, and the
retriever with the best comment-level retrieval is not the one with the best
answers. Do not claim an ordering.

**3. Retrieval buys 2.3×–3.2×, and that is the durable finding.** closed-book moves
from 0.221 to about 0.70 under the shipped rubric, and the gain survives two judges
(3.17× vs 2.33×), a rubric change, a 3.3× larger sample, and it is *larger* under
human review (~5×). Not "4×", and not 1.9× either — the interval moved up when the
sample grew, which is the opposite of the usual small-sample optimism.

**4. Retrieval makes the model confidently wrong rather than vague.** With
retrieval, fabrication sits at 20–28% and citation markers appear in 100% of
answers; without it, the model mostly admits ignorance and fabricates only 1.1% of
the time. Grounding and citation fidelity are separate problems and the second one
is unsolved here.

**5. Better retrieval does not automatically mean better answers.** The reranker
retrieves the answering comment most often (0.621 vs 0.611 for BM25) and still
scores lower end-to-end (0.679 vs 0.700). Whatever the generator does with the
window matters more than the last increment of retrieval quality — which is why
this repo reports the retrieval table *and* the answer table, and never derives one
from the other.


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
  to 0.084, the *same* judge run twice moves it by up to 0.026, and changing the
  rubric moved one system by 0.086. A blind human review of 30 pairs agrees with the
  judge on 70.0% of verdicts (quadratic kappa 0.816) and, more usefully, disagrees
  in a consistent direction: the judge over-credits vague advice by 0.100 and
  under-credits retrieval by 0.133. The remaining limit is *coverage* — one
  reviewer, 15 items per system, one sitting — not the absence of a check.
* **The issue-level retrieval metrics were a tautology, and are retired.** The
  original table reported recall@5 = 1.000 / MRR = 1.000 / leave-one-out = 0.000 for
  every retriever, on 29 items and on 95 items alike. Both halves are artefacts of
  how the task is built, not measurements:
  *the query is the issue title and every chunk is prefixed with that title, so the
  gold issue is retrieved at rank 1 by construction — measured 98–99.8% of each
  window is chunks of the gold issue itself;* and *the reference document belongs to
  the gold issue, so hiding that issue removes the only relevant document by
  definition, making leave-one-out identically zero.* The replacement in
  [Retrieval quality](#retrieval-quality--measured-on-the-answering-comment-not-the-issue)
  scores the answering **comment** instead, which does discriminate (0.516–0.621).
  The retired numbers are kept in `data/results/` as the archived arms, not as
  evidence.
* **Mined eval set, not a curated benchmark.** Maintainer replies are occasionally
  incomplete or answer a narrower question than the title suggests. The keep/drop
  review pass exists to cut those, and the sheet is committed so the selection is
  auditable: 200 candidates reviewed, 95 kept (47.5%).
* **One repository, one domain.** Conclusions about BM25-versus-dense weighting
  are domain-specific; a different preset will reorder the table.
* **Title-as-query favours lexical matching.** BM25 gets a slightly easier task
  than it would with paraphrased questions. With the comment-level metric in place
  this is now measurable rather than hidden, but a paraphrase-based query set is
  still the honest next step.
* **English-only retrieval.** The corpus is English issue text, the BM25 tokenizer
  applies English stopwords/stemming, and the embedding model is
  `bge-base-en-v1.5` — an English-only checkpoint. A question asked in another
  language is encoded into noise: retrieval returns unrelated threads and the
  generator correctly refuses instead of inventing an answer. That refusal is the
  right behaviour but a bad user experience, and it is a *design gap in this eval
  set*: all 95 items are English issue titles, so the pipeline was never measured
  on a non-English query. The demo page says "ask in English" for that reason.
  Fixing it properly means either a multilingual embedder (and re-baselining every
  number) or a translation step in front of retrieval (which would make the demo
  diverge from the measured pipeline — it is deliberately not shipped).
* **No multi-hop.** Every question is answered from a single issue thread. Questions
  that need two issues combined are out of scope.
* **Absolute scores are generator-bound.** The shipped runs use `deepseek-chat`
  through `--provider deepseek`; the default generator is a local 3B model, which
  scores lower in absolute terms. The closed-book vs RAG *gap* is the finding and it
  survives the stronger generator — the absolute numbers do not transfer.

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
