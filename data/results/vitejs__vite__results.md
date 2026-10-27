# Results — Vite (`vitejs/vite`)

- evaluation items: **3**
- generator: `deepseek-chat`
- judge: `deepseek-chat` (same model, greedy decoding)
- top-k retrieved: 5

## Answer quality

| system | accuracy | correct | partial | incorrect | fabricated | cited | p50 s | p95 s |
|---|---|---|---|---|---|---|---|---|
| `closed_book` | 0.667 | 66.7% | 0.0% | 33.3% | 0.0% | 0.0% | 4.28 | 4.4 |
| `bm25` | 1.000 | 100.0% | 0.0% | 0.0% | 0.0% | 100.0% | 3.61 | 3.87 |

_accuracy = correct(1.0) + partial(0.5) averaged; fabricated = judge flagged an invented version/API/path._

## Where retrieval actually matters (n=1)

Restricted to the items the closed-book model got **wrong**. On items the model already answers from memory, retrieval has nothing to add, and including them dilutes the comparison. This is a subgroup of the same run, not a filtered eval set: the closed-book row above still counts every item, so the full table is not cherry-picked.

| system | accuracy | correct | partial | incorrect | fabricated |
|---|---|---|---|---|---|
| `closed_book` | 0.000 | 0.0% | 0.0% | 100.0% | 0.0% |
| `bm25` | 1.000 | 100.0% | 0.0% | 0.0% | 0.0% |

## Retrieval (reference issue found in top-5)

| retriever | recall@5 | recall@5 leave-one-out | MRR | MRR LOO |
|---|---|---|---|---|
| `bm25` | 1.000 | 0.000 | 0.833 | 0.000 |

_leave-one-out hides the source issue's own documents, so it measures whether the retriever generalises rather than matching the query against itself._
