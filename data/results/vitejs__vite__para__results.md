# Results — Vite (`vitejs/vite`)

- evaluation items: **95**
- generator: `deepseek-chat`
- judge: `deepseek-chat` (same model, greedy decoding)
- top-k retrieved: 5

## Answer quality

| system | n | errors | accuracy | correct | partial | incorrect | fabricated | cited | p50 s | p95 s |
|---|---|---|---|---|---|---|---|---|---|---|
| `closed_book` | 95 | 0 | 0.289 | 9.5% | 38.9% | 51.6% | 6.3% | 0.0% | 4.27 | 5.16 |
| `bm25` | 95 | 0 | 0.674 | 56.8% | 21.1% | 22.1% | 17.9% | 97.9% | 3.72 | 51.43 |
| `dense` | 95 | 0 | 0.658 | 57.9% | 15.8% | 26.3% | 21.1% | 100.0% | 3.7 | 4.27 |
| `hybrid` | 95 | 0 | 0.668 | 58.9% | 15.8% | 25.3% | 20.0% | 100.0% | 3.79 | 4.4 |
| `hybrid_rerank` | 95 | 0 | 0.795 | 72.6% | 13.7% | 13.7% | 8.4% | 100.0% | 3.97 | 4.68 |

_accuracy = correct(1.0) + partial(0.5) averaged; fabricated = judge flagged an invented version/API/path; failed calls are counted in `errors` and excluded from accuracy._

## Where retrieval actually matters (n=87)

Restricted to the items the closed-book model got **wrong**. On items the model already answers from memory, retrieval has nothing to add, and including them dilutes the comparison. This is a subgroup of the same run, not a filtered eval set: the closed-book row above still counts every item, so the full table is not cherry-picked.

| system | accuracy | correct | partial | incorrect | fabricated |
|---|---|---|---|---|---|
| `closed_book` | 0.149 | 0.0% | 29.9% | 70.1% | 1.1% |
| `bm25` | 0.678 | 62.1% | 11.5% | 26.4% | 21.8% |
| `dense` | 0.621 | 54.0% | 16.1% | 29.9% | 28.7% |
| `hybrid` | 0.672 | 60.9% | 12.6% | 26.4% | 21.8% |
| `hybrid_rerank` | 0.672 | 60.9% | 12.6% | 26.4% | 24.1% |

## Retrieval (the maintainer's own comment found in top-5)

Relevance here is the reference **comment**, not the issue it came from. The issue-level version of this table is in Limitations because it is a tautology on this corpus.

| retriever | gold chunk@5 | gold chunk MRR | gold chunk top-1 | share of window from the same issue |
|---|---|---|---|---|
| `bm25` | 0.526 | 0.264 | 0.116 | 0.857 |
| `dense` | 0.558 | 0.242 | 0.084 | 0.973 |
| `hybrid` | 0.558 | 0.271 | 0.126 | 0.937 |
| `hybrid_rerank` | 0.716 | 0.359 | 0.158 | 0.922 |

_gold chunk@5 = the maintainer comment that answers the question appears among the retrieved chunks. The last column is the diagnosis of why the issue-level metric saturates: almost the whole window is chunks of the gold issue itself, so `recall@k` and MRR are 1.000 for every retriever by construction._

_Archived issue-level numbers (recall@5 / leave-one-out / MRR), kept only to show the degradation: `bm25` 0.968/0.000/0.935 · `dense` 1.000/0.000/0.984 · `hybrid` 0.990/0.000/0.981 · `hybrid_rerank` 1.000/0.000/0.979._
