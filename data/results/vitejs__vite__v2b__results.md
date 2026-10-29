# Results — Vite (`vitejs/vite`)

- evaluation items: **29**
- generator: `reused from the untagged run`
- judge: `deepseek-reasoner`
- top-k retrieved: 5

## Answer quality

| system | n | errors | accuracy | correct | partial | incorrect | fabricated | cited | p50 s | p95 s |
|---|---|---|---|---|---|---|---|---|---|---|
| `closed_book` | 29 | 0 | 0.035 | 0.0% | 6.9% | 89.7% | 93.1% | 0.0% | 4.31 | 4.89 |
| `hybrid_rerank` | 29 | 0 | 0.241 | 3.4% | 41.4% | 55.2% | 96.6% | 100.0% | 3.92 | 4.29 |

_accuracy = correct(1.0) + partial(0.5) averaged; fabricated = judge flagged an invented version/API/path; failed calls are counted in `errors` and excluded from accuracy._
