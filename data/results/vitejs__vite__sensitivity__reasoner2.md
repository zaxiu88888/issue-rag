# Grader sensitivity — Vite (`vitejs/vite`)

- baseline judge: `deepseek-reasoner [reasoner]`
- variant judge: `deepseek-reasoner [reasoner2]`
- answers are identical in both runs; only the grader differs (`evaluate --regrade`)

> **Both runs used the same model.** This table therefore measures the grader's own run-to-run noise, which sets the floor below which a difference between two systems must not be read as a result.

## Accuracy under each grader

| system | n | dropped | `deepseek-reasoner [reasoner]` | `deepseek-reasoner [reasoner2]` | delta |
|---|---|---|---|---|---|
| `closed_book` | 95 | 0 | 0.305 | 0.289 | -0.016 |
| `bm25` | 95 | 0 | 0.658 | 0.668 | +0.011 |
| `dense` | 95 | 0 | 0.595 | 0.621 | +0.026 |
| `hybrid` | 95 | 0 | 0.668 | 0.684 | +0.016 |
| `hybrid_rerank` | 95 | 0 | 0.711 | 0.705 | -0.005 |

## Do the two graders agree?

| system | n | exact | binary (usable vs not) |
|---|---|---|---|
| `closed_book` | 95 | 78.9% | 78.9% |
| `bm25` | 95 | 86.3% | 93.7% |
| `dense` | 95 | 84.2% | 90.5% |
| `hybrid` | 95 | 86.3% | 90.5% |
| `hybrid_rerank` | 95 | 80.0% | 89.5% |

## The claim that survives, and the one that does not

Best retrieval system by baseline accuracy: `hybrid_rerank`.

- gain under `deepseek-reasoner [reasoner]`: **+0.405** (0.305 -> 0.711)
- gain under `deepseek-reasoner [reasoner2]`: **+0.416** (0.289 -> 0.705)

**Ranking stability**: identical under both graders.

- under `deepseek-reasoner [reasoner]`: `closed_book < dense < bm25 < hybrid < hybrid_rerank`
- under `deepseek-reasoner [reasoner2]`: `closed_book < dense < bm25 < hybrid < hybrid_rerank`

- closed-book is the worst system under both graders: **True**
- the best system is the same under both: **True** (`hybrid_rerank`)

So the defensible claims are exactly those two, plus the gain interval. Any ordering between systems that swap places inside the noise band is unresolved, not a result.

## Verdict transitions

- `closed_book`: partial -> incorrect (12), incorrect -> partial (7), incorrect -> correct (1)
- `bm25`: partial -> correct (5), incorrect -> partial (3), correct -> partial (2), partial -> incorrect (2), correct -> incorrect (1)
- `dense`: partial -> correct (5), incorrect -> partial (4), partial -> incorrect (3), correct -> incorrect (1), correct -> partial (1), incorrect -> correct (1)
- `hybrid`: incorrect -> partial (7), correct -> partial (3), partial -> incorrect (2), partial -> correct (1)
- `hybrid_rerank`: partial -> correct (5), partial -> incorrect (5), correct -> partial (4), incorrect -> partial (3), incorrect -> correct (1), correct -> incorrect (1)
