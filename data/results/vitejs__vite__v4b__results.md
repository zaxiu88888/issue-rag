# Results — Vite (`vitejs/vite`)

- evaluation items: **29**
- generator: `reused from the untagged run`
- judge: `deepseek-reasoner`
- top-k retrieved: 5

## Answer quality

| system | n | errors | accuracy | correct | partial | incorrect | fabricated | cited | p50 s | p95 s |
|---|---|---|---|---|---|---|---|---|---|---|
| `closed_book` | 29 | 0 | 0.259 | 3.4% | 44.8% | 48.3% | 0.0% | 0.0% | 4.31 | 4.89 |
| `hybrid_rerank` | 29 | 0 | 0.552 | 37.9% | 34.5% | 27.6% | 20.7% | 100.0% | 3.92 | 4.29 |

_accuracy = correct(1.0) + partial(0.5) averaged; fabricated = judge flagged an invented version/API/path; failed calls are counted in `errors` and excluded from accuracy._
