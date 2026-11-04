# Paraphrased queries — does the title advantage hold?

- evaluation items: **95**
- corpus: `data/processed/vitejs__vite/corpus.jsonl`
- paraphrases: `data/processed/vitejs__vite/paraphrases.jsonl` (one rewrite per item)
- relevance: the maintainer comment that answers the question (gold chunk)

| retriever | gold chunk@5 title | gold chunk@5 paraphrase | Δ | MRR title | MRR paraphrase | Δ |
|---|---|---|---|---|---|---|
| `bm25` | 0.611 | 0.526 | -0.084 | 0.280 | 0.264 | -0.016 |
| `dense` | 0.516 | 0.558 | +0.042 | 0.206 | 0.242 | +0.036 |
| `hybrid` | 0.579 | 0.558 | -0.021 | 0.247 | 0.271 | +0.024 |
| `hybrid_rerank` | 0.621 | 0.716 | +0.095 | 0.289 | 0.359 | +0.070 |

Best retriever by gold chunk@5: **hybrid_rerank** on titles, **hybrid_rerank** on paraphrases.

## How to read this

Titles are close to the wording the maintainer used, so a lexical retriever gets an
easier case; that was an admitted bias with no measurement behind it until this run.
A negative Δ means the paraphrase removed some of that advantage.

The number to watch is the **relative** drop: if `bm25` falls further than `dense`,
the title advantage was real and the BM25 rows in the main table are optimistic.
