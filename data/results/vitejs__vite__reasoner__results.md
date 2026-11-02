# Results — Vite (`vitejs/vite`)

- evaluation items: **95**
- generator: `reused from the untagged run`
- judge: `deepseek-reasoner`
- top-k retrieved: 5

## Answer quality

| system | n | errors | accuracy | correct | partial | incorrect | fabricated | cited | p50 s | p95 s |
|---|---|---|---|---|---|---|---|---|---|---|
| `closed_book` | 95 | 0 | 0.305 | 7.4% | 46.3% | 46.3% | 3.2% | 0.0% | 4.32 | 5.0 |
| `bm25` | 95 | 0 | 0.658 | 56.8% | 17.9% | 25.3% | 23.2% | 100.0% | 3.7 | 4.29 |
| `dense` | 95 | 0 | 0.595 | 48.4% | 22.1% | 29.5% | 26.3% | 100.0% | 3.65 | 4.29 |
| `hybrid` | 95 | 0 | 0.668 | 60.0% | 13.7% | 26.3% | 16.8% | 100.0% | 3.53 | 4.19 |
| `hybrid_rerank` | 95 | 0 | 0.711 | 60.0% | 22.1% | 17.9% | 14.7% | 100.0% | 3.77 | 4.33 |

_accuracy = correct(1.0) + partial(0.5) averaged; fabricated = judge flagged an invented version/API/path; failed calls are counted in `errors` and excluded from accuracy._
