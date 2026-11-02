# Results — Vite (`vitejs/vite`)

- evaluation items: **95**
- generator: `reused from the untagged run`
- judge: `deepseek-reasoner`
- top-k retrieved: 5

## Answer quality

| system | n | errors | accuracy | correct | partial | incorrect | fabricated | cited | p50 s | p95 s |
|---|---|---|---|---|---|---|---|---|---|---|
| `closed_book` | 95 | 0 | 0.289 | 8.4% | 41.1% | 50.5% | 3.2% | 0.0% | 4.32 | 5.0 |
| `bm25` | 95 | 0 | 0.668 | 58.9% | 15.8% | 25.3% | 22.1% | 100.0% | 3.7 | 4.29 |
| `dense` | 95 | 0 | 0.621 | 52.6% | 18.9% | 28.4% | 22.1% | 100.0% | 3.65 | 4.29 |
| `hybrid` | 95 | 0 | 0.684 | 57.9% | 21.1% | 21.1% | 15.8% | 100.0% | 3.53 | 4.19 |
| `hybrid_rerank` | 95 | 0 | 0.705 | 61.1% | 18.9% | 20.0% | 16.8% | 100.0% | 3.77 | 4.33 |

_accuracy = correct(1.0) + partial(0.5) averaged; fabricated = judge flagged an invented version/API/path; failed calls are counted in `errors` and excluded from accuracy._
