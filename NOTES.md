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
