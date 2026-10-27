# Results — Vite (`vitejs/vite`)

- evaluation items: **29**
- generator: `deepseek-chat`
- judge: `deepseek-chat` (same model, greedy decoding)
- top-k retrieved: 5

## Answer quality

| system | accuracy | correct | partial | incorrect | fabricated | cited | p50 s | p95 s |
|---|---|---|---|---|---|---|---|---|
| `closed_book` | 0.138 | 3.4% | 20.7% | 75.9% | 13.8% | 0.0% | 4.31 | 4.89 |
| `bm25` | 0.483 | 31.0% | 34.5% | 34.5% | 31.0% | 100.0% | 3.76 | 4.45 |

_accuracy = correct(1.0) + partial(0.5) averaged; fabricated = judge flagged an invented version/API/path._

## Where retrieval actually matters (n=28)

Restricted to the items the closed-book model got **wrong**. On items the model already answers from memory, retrieval has nothing to add, and including them dilutes the comparison. This is a subgroup of the same run, not a filtered eval set: the closed-book row above still counts every item, so the full table is not cherry-picked.

| system | accuracy | correct | partial | incorrect | fabricated |
|---|---|---|---|---|---|
| `closed_book` | 0.107 | 0.0% | 21.4% | 78.6% | 14.3% |
| `bm25` | 0.464 | 28.6% | 35.7% | 35.7% | 32.1% |

## Retrieval (reference issue found in top-5)

| retriever | recall@5 | recall@5 leave-one-out | MRR | MRR LOO |
|---|---|---|---|---|
| `bm25` | 1.000 | 0.000 | 1.000 | 0.000 |

_leave-one-out hides the source issue's own documents, so it measures whether the retriever generalises rather than matching the query against itself._
