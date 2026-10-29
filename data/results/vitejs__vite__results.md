# Results — Vite (`vitejs/vite`)

- evaluation items: **29**
- generator: `deepseek-chat`
- judge: `deepseek-chat` (same model, greedy decoding)
- top-k retrieved: 5

## Answer quality

| system | n | errors | accuracy | correct | partial | incorrect | fabricated | cited | p50 s | p95 s |
|---|---|---|---|---|---|---|---|---|---|---|
| `closed_book` | 29 | 0 | 0.190 | 10.3% | 17.2% | 72.4% | 3.4% | 0.0% | 4.31 | 4.89 |
| `bm25` | 29 | 0 | 0.517 | 37.9% | 27.6% | 34.5% | 34.5% | 100.0% | 3.76 | 4.45 |
| `dense` | 29 | 0 | 0.466 | 27.6% | 37.9% | 34.5% | 24.1% | 100.0% | 3.74 | 4.3 |
| `hybrid` | 29 | 0 | 0.500 | 37.9% | 24.1% | 37.9% | 37.9% | 100.0% | 3.68 | 3.97 |
| `hybrid_rerank` | 29 | 0 | 0.500 | 44.8% | 10.3% | 44.8% | 34.5% | 100.0% | 3.92 | 4.29 |

_accuracy = correct(1.0) + partial(0.5) averaged; fabricated = judge flagged an invented version/API/path; failed calls are counted in `errors` and excluded from accuracy._

## Where retrieval actually matters (n=26)

Restricted to the items the closed-book model got **wrong**. On items the model already answers from memory, retrieval has nothing to add, and including them dilutes the comparison. This is a subgroup of the same run, not a filtered eval set: the closed-book row above still counts every item, so the full table is not cherry-picked.

| system | accuracy | correct | partial | incorrect | fabricated |
|---|---|---|---|---|---|
| `closed_book` | 0.096 | 0.0% | 19.2% | 80.8% | 3.8% |
| `bm25` | 0.481 | 34.6% | 26.9% | 38.5% | 34.6% |
| `dense` | 0.442 | 26.9% | 34.6% | 38.5% | 26.9% |
| `hybrid` | 0.519 | 38.5% | 26.9% | 34.6% | 34.6% |
| `hybrid_rerank` | 0.500 | 46.2% | 7.7% | 46.2% | 34.6% |

## Retrieval (reference issue found in top-5)

| retriever | recall@5 | recall@5 leave-one-out | MRR | MRR LOO |
|---|---|---|---|---|
| `bm25` | 1.000 | 0.000 | 1.000 | 0.000 |
| `dense` | 1.000 | 0.000 | 1.000 | 0.000 |
| `hybrid` | 1.000 | 0.000 | 1.000 | 0.000 |
| `hybrid_rerank` | 1.000 | 0.000 | 1.000 | 0.000 |

_leave-one-out hides the source issue's own documents, so it measures whether the retriever generalises rather than matching the query against itself._
