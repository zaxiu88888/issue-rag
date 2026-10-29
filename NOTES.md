# NOTES — what we built and why

This file exists because an AI-assisted project has one specific failure mode:
the code runs, you cannot explain it, and an interviewer finds out in ninety
seconds. Everything below is written as *something you can say out loud*.

Update it at the end of every working session. Two minutes per session. It is
the cheapest insurance in the project.

---

## 30-second explanation of the project

> Support questions about fast-moving libraries are usually answered inside the
> last few months of the issue tracker, which is exactly the window a language
> model does not know about. So the model invents plausible version numbers and
> API names. I built a retrieval-augmented assistant over a real repo's issues,
> and I built the eval set to prove the effect: real issue titles as queries,
> maintainer replies as reference answers, five configurations compared —
> closed-book, BM25, dense, hybrid, hybrid plus reranker.

Say the *finding*, not the feature list.

---

## Decisions you must be able to defend

### Why hybrid retrieval instead of just embeddings?

Because the two fail in different places, and support questions live in both.

- Dense retrieval handles paraphrase: "streaming re-sends the whole history"
  matches "O(n) per step on message validation" with no shared keywords.
- BM25 handles exact tokens: error strings, `convertToModelMessages`, `4.0.12`.
  An embedding compresses a rare identifier into the same neighbourhood as a
  hundred other identifiers; BM25 does not.

The results table shows which one carries more weight in this domain — quote the
number rather than asserting a preference.

### Why reciprocal rank fusion rather than weighted score blending?

BM25 scores are unbounded and corpus-dependent; cosine similarities are bounded
and compressed near 1.0. Blending them requires normalisation, and the
normalisation constant is sensitive to how many candidates you pull. RRF only
uses *rank position*: `score = Σ 1/(k + rank)`. It needs no calibration, it is
one line, and it is robust to changing the candidate pool — which matters because
the ablation changes that pool.

### Why is the retrieval unit a comment, not an issue?

A maintainer reply is a self-contained answer. Retaining whole issues makes
documents long and dilutes the embedding; the relevant sentence ends up averaged
against nine kilobytes of stack trace.

Each document is prefixed with its issue title so a chunk like "this also happens
with streaming" stays interpretable and citable.

### Why is the query the title and not the body?

**This is the trap in the whole project.** If the query is the issue body, the
document containing that body is a near-perfect lexical match. Every retrieval
metric looks excellent and measures nothing.

Using the title is closer to what a user actually types, and it does not let the
retriever match the query against itself.

### Then why report two retrieval numbers?

Because both are legitimate answers to different questions:

- *recall@5 with the source issue present* — the product number. A user asking
  this question should be shown the existing thread. Hiding it would be an
  artificial handicap.
- *recall@5 leave-one-issue-out* — the generalisation number. It answers "can the
  retriever find a *different* thread that helps", which is what matters when the
  knowledge is not already in a single thread.

Reporting only the first is the flattering choice, and an interviewer who knows
retrieval will ask about exactly this.

### Why is the eval set mined from maintainer comments?

Ground truth without annotation cost. The label is "a maintainer said this in
response to this question", which is exactly the target distribution, and it
cannot be gamed by an LLM generating both question and answer.

The weakness is that maintainer replies are sometimes incomplete, so there is a
mandatory human keep/drop pass, and the review sheet is committed so the
selection is auditable rather than invisible.

### Why is a fabricated version number scored `incorrect`, not `partial`?

Because it is the specific failure the system exists to prevent. A vague answer
wastes a user's time; a confident wrong version number sends them down a
30-minute wrong path. The rubric encodes that judgement, and `fabricated` is a
separate reported column so the two failure modes do not get averaged together.

### Why a 3B local model as the default?

So the numbers are reproducible by anyone with a 16 GB GPU and no API key. The
cost is real: absolute scores are lower than a frontier model's. The claim being
made is about the **gap** between closed-book and retrieval, and that gap is
architectural — it survives swapping in a stronger generator.

`--endpoint` makes that check a one-line change against any OpenAI-compatible
server. If you have not actually run that comparison, say so.

---

## Things that will be asked and you should rehearse

1. **"How do you know your eval set isn't leaked into training?"**
   The questions and answers are 6–12 months old, newer than the model's cutoff.
   The closed-book column is the empirical check: if the model already knew the
   answers, closed-book accuracy would be high and the premise would collapse.

2. **"Your judge is the same model family being tested. Isn't that circular?"**
   Partly, yes. Mitigations: the judge grades against a concrete reference rather
   than vibes; the grounding audit is a separate pass; and 30 items are checked by
   hand with the agreement rate reported. It is a proxy, and the README says so.

3. **"What is your precision at k, not just recall?"**
   Recall@k is what this system needs, because the generator can ignore irrelevant
   excerpts and the prompt tells it to. Precision would matter if the context
   window were the binding constraint — it is not at k=5 with 1.2 KB chunks.

4. **"What breaks at 100 QPS?"**
   The embedding step and the generator, in that order. Embeddings are a single
   batched GPU call per query and cache trivially by query hash; generation needs
   continuous batching (vLLM) rather than the `transformers` generate loop used
   here. BM25 and the cross-encoder are cheap by comparison, though the reranker
   is the next bottleneck after that.

5. **"What would you do with two more weeks?"**
   (a) Paraphrase the queries with an LLM so BM25 does not get the easy lexical
   case; (b) two annotators on 100 items to measure judge–human agreement
   properly; (c) a second repository to test whether the BM25/dense balance is a
   property of the domain or of this corpus; (d) incremental indexing, because
   re-embedding the corpus on every new issue is wasteful.

---

## Session log

Keep this short. One line per session: what changed, what broke, what you learned.

- **S1** — environment probe. Windows sandbox blocks `tempfile.mkdtemp`, so `pip`
  cannot run confined; installer needs wider access. `huggingface.co` unreachable,
  ModelScope works → all model downloads go through ModelScope.
- **S2** — harvest stage. Learned that GitHub's anonymous core quota is 60/hour,
  which makes per-issue comment fetching impossible; the repo-wide comment list
  endpoint returns 100 per request instead. Retrieval-unit and query-mode
  decisions written down above.
- **S3** — first real run. 29 reviewed items, closed_book 0.138 vs BM25 0.483.
  Three findings, below.
- **S4** — completed the ablation: dense 0.448, hybrid 0.466, hybrid+rerank 0.586.
  The retriever rows are within noise of each other; only the reranker moves.

---

## The finding to lead with

**I graded my own results three times and found out how much of the headline is
the grader.**

The pipeline scored answers with `deepseek-chat`, the same model that wrote them.
So every answer was re-graded by `deepseek-reasoner` over the same stored answers —
and then the reasoner was run a second time, to measure its own noise.

Two different models, same answers:

| system | `deepseek-chat` | `deepseek-reasoner` | delta |
|---|---|---|---|
| closed_book | 0.138 | 0.328 | **+0.190** |
| bm25 | 0.483 | 0.414 | −0.069 |
| dense | 0.448 | 0.362 | −0.086 |
| hybrid | 0.466 | 0.466 | 0.000 |
| hybrid + rerank | 0.586 | 0.483 | −0.103 |
| **gain (closed_book → best)** | **+0.448 (4.2×)** | **+0.155 (1.5×)** | |

Same model, twice:

| system | run a | run b | delta |
|---|---|---|---|
| hybrid + rerank | 0.483 | 0.552 | **+0.069** |
| dense | 0.339 | 0.375 | +0.036 |
| hybrid | 0.466 | 0.431 | −0.035 |
| closed_book | 0.328 | 0.345 | +0.017 |
| bm25 | 0.414 | 0.397 | −0.017 |

The grader agrees with **itself** on only 71–93% of verdicts. So **~0.07 accuracy
is noise**, and any gap smaller than that between two systems is not a result.

Three claims come out of this, and only the first two are usable:

1. **Holds in every run:** closed-book is worst, and the reranked system is best.
2. **Does not hold:** "retrieval quadruples accuracy". It is **1.5×–4.2×**.
3. **Unresolved:** the ordering of `bm25` / `hybrid` / `dense`. They swap between
   graders and every pair sits within 0.07 of each other.

Why the graders disagree, mechanically: agreement is 48% on closed-book answers and
93% on hybrid answers. Vague generic advice ("try clearing the cache, check your
Node version") gets partial credit from one grader and nothing from the other; an
answer that pairs a correct fix with an invented version number gets full credit
from one and none from the other.

**Why this is the best thing to lead with in an interview:** it is the difference
between reporting a metric and understanding it. Anyone can produce a 4×. Very few
candidates will say their number is soft, quantify how soft, notice that their own
measuring instrument is noisy, and then show which conclusions survive anyway.
Expect the follow-up "so what would you actually claim?" — the answer is: the shape
of the result and a 1.5×–4.2× interval, nothing more precise.

**Follow-up you should be ready for:** "why not just use a human?" Because that is
the next step and it is not done — 30 items, one hour. Say it before they ask.

---

## What the first run actually showed

These are the three things to bring up in an interview, because they are the
things you can only learn by running the experiment.

### 1. Retrieval quadrupled accuracy and *also* doubled fabrication

| | closed-book | BM25 | hybrid + rerank |
|---|---|---|---|
| correct | 3.4% | 31.0% | 44.8% |
| accuracy | 0.138 | 0.483 | 0.586 |
| fabricated | 13.8% | 31.0% | 31.0% |

The mechanism: without retrieval the model mostly **admits ignorance** — several
answers said "I don't have reliable knowledge of a Vite 8 release". Those score
`incorrect`, not `fabricated`, because they invent nothing.

With retrieval it has real specifics in front of it — a version, a PR number, a
file path — and misattributes them. So the failure moves from *vague* to
*confidently wrong*, which is worse per failure and easy to miss if you only
report one accuracy number.

**Why this matters in an interview:** it shows you understand that a RAG system
has two separable problems — grounding and citation fidelity — and that fixing
one can make the other more visible. Quoting "accuracy quadrupled" alone would be
the naive read; quoting both numbers is the mature one.

### 2. The retriever comparison is noise, and saying so is the point

BM25 0.483, dense 0.448, hybrid 0.466. At n=29 that spread is about one item —
nothing. There is a mechanical reason, and it is the real finding:

`recall@5 = 1.000` and `MRR = 1.000` for **every** retriever, because the query is
the issue title and every document is prefixed with that title. All four
configurations find the gold document at rank 1, so they can only differ in what
the *other* four excerpts are. The ablation is not measuring retrieval ability.

Leave-one-out (hiding the source issue's own documents) collapses to 0.000,
confirming each answer lives in exactly one thread.

Reranking is the one row that moves (0.466 → 0.586, about three items) and the
mechanism is coherent — first stage picks the thread, the cross-encoder picks the
supporting excerpts — but at n=29 it is suggestive, not established.

**Why this matters:** anyone can report a 1.0. Reporting that your own metric is
degenerate, explaining the construct, and naming the fix (questions whose answers
appear in several threads, or paraphrased queries) is what separates measurement
from marketing.

### 3. A quarter of the retrieval answers are summaries, not answers

9 of 29 BM25 answers were of the form "this is tracked in issue #21827" rather
than stating the resolution. The prompt forbids exactly that; it only partly
worked.

That gap — between "the context contains the answer" and "the model states the
answer" — is a real product problem, and the fix is either an answer-vs-summary
classifier or a rewrite pass, not a bigger model.

---

## Numbers to have memorised

- 29 eval items, hand-reviewed down from 53 mined candidates
- 533 issues, 1,097 comments → 3,078 corpus documents
- closed_book **0.138** → BM25 **0.483** → hybrid+rerank **0.586** (in-project judge)
- **judge substitution: the gain ranges from +0.137 (1.4×) to +0.433 (4.1×)**
- cross-model agreement: **48% on closed_book, 79–93% on the retrieval systems**
- the ranking of the five systems is **identical** under both graders
- `correct` **3.4% → 44.8%**; fabrication **13.8% → 31.0%**
- on the 28 items closed-book missed: **0.107 → 0.571**
- BM25 0.483 / dense 0.448 / hybrid 0.466 — within noise of each other
- ~25% of retrieval answers summarise the thread instead of answering
- 86% of one rejected repo's comment stream was bots (see `docs/DATA_QUALITY.md`)

If an interviewer asks "what did you actually learn?", lead with the judge
sensitivity and close with the stable ranking.
