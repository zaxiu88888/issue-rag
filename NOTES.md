# NOTES — what we built and why

**English** | [简体中文](NOTES.zh-CN.md)

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
- **S5** — widened the sample: 18-month window, 771 issues / 2,593 comments /
  5,473 corpus documents, 200 candidates reviewed by hand down to **95 items**
  (47.5% keep rate). Re-ran the whole matrix. closed_book 0.221, bm25 0.700,
  dense 0.632, hybrid 0.695, rerank 0.679 — the gain grew from 2.72× to 3.17×.

---

## The finding to lead with

**I audited my own measuring instrument three ways, and it changed my headline.**

The pipeline scored answers with `deepseek-chat`, the same model that wrote them.
So the questions asked were, in order: does the *rubric* matter, does the *judge
model* matter, and does the judge even agree with itself? All comparisons reuse the
same stored answers (`evaluate --regrade`), so only the grader varies.

**a) Judge model** (rubric ③, 95 items): `closed_book` 0.221 → 0.305, reranked
0.679 → 0.711 under `deepseek-reasoner`. Gain 3.17× vs 2.33×. The stricter grader is
*lenient* on closed-book and *strict* on retrieval, which compresses the gain.

**b) Same judge, twice** (rubric ③, 95 items): run-to-run difference 0.005–0.026 on
the aggregate, exact agreement 79–86%. The ordering is identical in both runs.

**c) The rubric itself** — this is the one that mattered. Same model, same answers,
rubric ① vs ③ (measured on the 29-item set, because rubric ① no longer exists in the
code and re-implementing it would change the instrument under test):

| system | rubric ① | rubric ③ | delta |
|---|---|---|---|
| closed_book | 0.138 | 0.190 | +0.052 |
| bm25 | 0.483 | 0.517 | +0.035 |
| dense | 0.448 | 0.466 | +0.017 |
| hybrid | 0.466 | 0.500 | +0.035 |
| **hybrid + rerank** | **0.586** | **0.500** | **−0.086** |

**d) Does a human agree?** 30 answers graded blind through a local page: the
reviewer saw question, reference and candidate, never the verdict, never the system.

| metric | value |
|---|---|
| exact agreement | 70.0% |
| usable-vs-not | 73.3% |
| quadratic-weighted kappa | **0.816** (almost perfect) |

| system | n | mean human | mean judge |
|---|---|---|---|
| closed_book | 15 | **0.167** | 0.267 |
| hybrid + rerank | 15 | **0.867** | 0.733 |

**This is the answer to "isn't your judge circular?"** — and it points the
reassuring way: the judge is *lenient* on vague advice (+0.100) and *strict* on real
answers (−0.133), so the machine-measured gain is a **lower bound**. On the sampled
items the human sees +0.700 (5.2×) where the judge sees +0.466 (2.75×).

Say the caveat before you are asked: one reviewer, 15 per system, one sitting. It
validates the direction, not the decimal. The sample-size experiment points the same
way: tripling the eval set moved closed_book by only +0.031 while retrieval moved
by ~0.18, so both independent checks say the measured gain is a floor, not a ceiling.

Three conclusions, and only the first is a keeper in its original form:

1. **Holds everywhere:** retrieval beats closed-book. Every rubric, every judge,
   every run, every sample size. 0.221 → about 0.70.
2. **Withdrawn:** "the reranked system is best". It led by 0.10 under rubric ①, ties
   or trails under ③, and the ordering flips between judges. A result that flips
   when the instrument is corrected was a result about the instrument.
3. **Narrowed, then widened again:** the gain was "4×", then 1.5×–4.2×, then
   **1.9×–2.6×** after the rubric fix — and after tripling the eval set it is
   **2.3×–3.2×**. The first move was the instrument getting better; the second was
   the sample getting big enough to see the effect more clearly. Both directions
   belong in the story, and reporting the second one is what makes the first one
   credible.

**Why this is the best thing to lead with in an interview:** anyone can produce a
4×. Very few candidates will say their number is soft, quantify how soft, discover
that their own rubric was manufacturing the headline, and then withdraw a claim.
Expect "so what would you actually claim?" — the answer is: retrieval helps by
**2.3×–3.2×**, the four retrieval configurations are indistinguishable, and here is
the interval.

**Follow-up you should be ready for:** "why not just use a human?" Because that is
the next step and it is not done — 30 items, one hour. Say it before they ask.

---

## The sequel: I made the grader more reproducible and it got worse

Having measured that the grader was noisy (0.07 floor), the obvious next move was
to remove the noise. The noise lived on the partial/incorrect boundary, so I took
that decision away from the model: it now answers four factual sub-questions and
**code** maps those answers to a verdict. Same questions, deterministic mapping,
nothing left to drift.

It worked, in the sense that agreement rose from 82.8% to 93.1%. It also destroyed
both scores — closed_book 0.336 → 0.034, reranked 0.517 → 0.259 — because one of
the four checks, `wrong_specific`, was defined as "names any version, issue, file
path, config key or API symbol absent from the reference". A retrieval answer
legitimately quotes details from *other* excerpts, so 93–97% of answers were
flagged as fabricating.

Narrowing that check to version and issue numbers that actually conflict with the
reference fixed it:

| rubric | closed_book | reranked | run-to-run Δ (reranked) | exact agreement |
|---|---|---|---|---|
| holistic (original) | 0.336 | 0.517 | 0.069 | 82.8% |
| four checks, broad | 0.034 | 0.259 | 0.069 | **93.1%** |
| four checks, narrow | 0.287 | **0.562** | **0.018** | 82.1% |

**The lesson is the interview answer.** "Better" for a grader is not agreement.
It is all three of: run-to-run difference on the number you report, the level
itself, and which sub-answer fired. The broad rubric won on the first and lost on
the other two, and a candidate who only tracked agreement would have shipped it.

Note the last row honestly: the narrow rubric does **not** improve per-item
agreement (82.1% vs 82.8%). It improves the *net* shift between runs — errors
cancel instead of biasing the aggregate. Since every published number is an
aggregate, that is the variance that mattered.

---

## What the shipped run shows

These are the three things to bring up in an interview, because they are the
things you can only learn by running the experiment. Numbers are the shipped
rubric ③.

### 1. Retrieval roughly triples accuracy and *multiplies* fabrication

| | closed-book | best retrieval config |
|---|---|---|
| correct | 8.4% | 64.2% |
| accuracy | 0.221 | 0.700 |
| fabricated | 1.1% | 20.0% |

The mechanism: without retrieval the model mostly **admits ignorance** — several
answers said "I don't have reliable knowledge of a Vite 8 release". Those score
`incorrect`, not `fabricated`, because they invent nothing. With retrieval it has
real specifics in front of it — a version, a PR number, a file path — and
misattributes them. The failure moves from *vague* to *confidently wrong*, which is
worse per failure and easy to miss if you report one accuracy number.

**Why this matters in an interview:** it shows you understand that a RAG system has
two separable problems — grounding and citation fidelity — and that fixing one can
make the other more visible.

### 2. The four retrieval configurations are indistinguishable — *on title queries*

BM25 0.700, hybrid 0.695, rerank 0.679, dense 0.632. That is a 0.07 spread, and
four independent lines say it is not a ranking:

1. **It is at or inside the instrument's own noise.** Same-judge run-to-run is
   0.005–0.026; changing the judge moves systems by up to 0.084 and *flips the
   ordering* (bm25 first under `deepseek-chat`, rerank first under `deepseek-reasoner`).
2. **The ordering is not stable across graders**, which is the cheapest possible
   falsification test and it fails.
3. **The reranker's lead did not survive a rubric fix.** It won by 0.10 under
   rubric ① and lost it under ③.
4. **The retriever that retrieves best does not answer best.** On the comment-level
   metric the reranker leads (0.621 vs BM25's 0.611) while trailing on accuracy
   (0.679 vs 0.700).

That third point is the one to say out loud, because it is a claim I had to
withdraw rather than a claim I never made.

**But there is a fifth line now, and it flips the headline.** Rewrite the 95 titles
as user questions and re-run the answers, and the ordering changes:

| system | title | paraphrase | Δ |
|---|---|---|---|
| bm25 | **0.700** | 0.674 | −0.026 |
| hybrid_rerank | 0.679 | **0.795** | **+0.116** |

The reranker wins by 0.12 under user-voiced queries, and the closed-book row rises
(0.221 → 0.289) because a paraphrase spells out the problem more explicitly. So the
honest claim is: **"indistinguishable" holds when the query echoes the corpus
wording (titles); when a user asks in their own words, the reranker separates and
the four-stop ordering is real.** The ranking is not a property of the retrievers
alone — it is a property of the query, and that is the most transferable thing in
this whole repo.

**Why this matters:** anyone can report a 1.0. Reporting that your own metric is
degenerate, explaining the construct, and then *replacing* it — the issue-level
`recall@5` was a tautology (98–99.8% of each window was chunks of the gold issue
itself, and leave-one-out was zero by definition), so it is now scored on the
answering comment — is what separates measurement from marketing. Then re-running
the *whole thing* on paraphrased queries and watching the ranking flip is what
separates an eval from a screenshot.

### 3. Better retrieval is not the same as better answers

The reranker puts the answering comment in the top 5 more often than any other
retriever (62.1% vs 51.6–61.1%) and still finishes behind BM25 end-to-end. The gap
between "the context contains the answer" and "the model states the answer" is a
real product problem, and the fix is an answer-vs-summary check or a rewrite pass,
not a bigger model.

---

## Numbers to have memorised

- **95 eval items**, hand-reviewed down from 200 mined candidates (47.5% keep rate)
- 771 issues, 2,593 comments → **5,473 corpus documents**
- closed_book **0.221**; retrieval configurations **0.632–0.700** (shipped rubric ③, title queries)
- **paraphrased queries flip the ranking**: rerank 0.679 → **0.795**, bm25 0.700 → 0.674,
  closed_book 0.221 → 0.289 — "indistinguishable" holds only for title queries
- gain **2.3×–3.2×** across two judges: 3.17× (chat) vs 2.33× (reasoner)
- `correct` **8.4% → 64.2%**; fabrication **1.1% → 20–28%**
- on the 87 items closed-book missed: **0.149 → 0.621–0.678**
- noise floor: same-judge run-to-run 0.005–0.026; a judge swap moves one system by
  0.084 and reorders the table. Anything under ~0.03 is not a result.
- **the reranker's lead was withdrawn** — 0.586 under rubric ①, 0.500 under ③
- **sample size**: 29 → 95 items moved closed_book by only **+0.031** while
  retrieval moved ~0.18, so the gain grew (2.72× → 3.17×), it did not regress
- **human blind review: 70.0% exact, κ=0.816, and the judge is lenient on vague
  advice (+0.100) and strict on good answers (−0.133)** — so the machine numbers
  understate the gain (human sees 5.2×, judge sees 2.75×)
- **retrieval metric**: issue-level recall/MRR were 1.000/1.000 for everyone by
  construction; comment-level gold-chunk@5 is 0.516–0.621 and does rank them
- 86% of one rejected repo's comment stream was bots (see `docs/DATA_QUALITY.md`)

If an interviewer asks "what did you actually learn?", lead with the instrument
audit — the rubric manufactured the headline, and finding that out narrowed the
claim from "4×" to "about 2–3×" without anyone adding a single data point. Then
follow with the sequel: tripling the sample moved the control barely at all but
moved the treatment a lot, which is what you want to see and is rarely checked.
