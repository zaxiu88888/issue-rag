# Results — Vite (`vitejs/vite`)

- evaluation items: **95**
- generator: `reused from the untagged run`
- judge: `deepseek-chat`
- top-k retrieved: 5

## Answer quality

| system | n | errors | accuracy | correct | partial | incorrect | fabricated | cited | p50 s | p95 s |
|---|---|---|---|---|---|---|---|---|---|---|
| `closed_book` | 95 | 0 | 0.174 | 6.3% | 22.1% | 71.6% | 7.4% | 0.0% | 4.32 | 5.0 |
| `bm25` | 95 | 0 | 0.737 | 66.3% | 14.7% | 18.9% | 41.1% | 100.0% | 3.7 | 4.29 |
| `dense` | 95 | 0 | 0.695 | 58.9% | 21.1% | 20.0% | 42.1% | 100.0% | 3.65 | 4.29 |
| `hybrid` | 95 | 0 | 0.716 | 62.1% | 18.9% | 18.9% | 38.9% | 100.0% | 3.53 | 4.19 |
| `hybrid_rerank` | 95 | 0 | 0.768 | 68.4% | 16.8% | 14.7% | 34.7% | 100.0% | 3.77 | 4.33 |

_accuracy = correct(1.0) + partial(0.5) averaged; fabricated = judge flagged an invented version/API/path; failed calls are counted in `errors` and excluded from accuracy._
