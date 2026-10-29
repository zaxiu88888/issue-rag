# Results — Vite (`vitejs/vite`)

- evaluation items: **29**
- generator: `reused from the untagged run`
- judge: `deepseek-reasoner`
- top-k retrieved: 5

## Answer quality

| system | n | errors | accuracy | correct | partial | incorrect | fabricated | cited | p50 s | p95 s |
|---|---|---|---|---|---|---|---|---|---|---|
| `closed_book` | 29 | 0 | 0.345 | 3.4% | 62.1% | 34.5% | 20.7% | 0.0% | 4.31 | 4.89 |
| `bm25` | 29 | 0 | 0.397 | 24.1% | 31.0% | 44.8% | 44.8% | 100.0% | 3.76 | 4.45 |
| `dense` | 29 | 0 | 0.362 | 20.7% | 31.0% | 44.8% | 37.9% | 100.0% | 3.74 | 4.3 |
| `hybrid` | 29 | 0 | 0.431 | 27.6% | 31.0% | 41.4% | 34.5% | 100.0% | 3.68 | 3.97 |
| `hybrid_rerank` | 29 | 0 | 0.552 | 41.4% | 27.6% | 31.0% | 31.0% | 100.0% | 3.92 | 4.29 |

_accuracy = correct(1.0) + partial(0.5) averaged; fabricated = judge flagged an invented version/API/path; failed calls are counted in `errors` and excluded from accuracy._
