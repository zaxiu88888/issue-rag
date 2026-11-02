# Grader sensitivity — Vite (`vitejs/vite`)

- baseline judge: `deepseek-chat [baseline]`
- variant judge: `deepseek-reasoner [reasoner]`
- answers are identical in both runs; only the grader differs (`evaluate --regrade`)

## Accuracy under each grader

| system | n | dropped | `deepseek-chat [baseline]` | `deepseek-reasoner [reasoner]` | delta |
|---|---|---|---|---|---|
| `closed_book` | 95 | 0 | 0.221 | 0.305 | +0.084 |
| `bm25` | 95 | 0 | 0.700 | 0.658 | -0.042 |
| `dense` | 95 | 0 | 0.632 | 0.595 | -0.037 |
| `hybrid` | 95 | 0 | 0.695 | 0.668 | -0.026 |
| `hybrid_rerank` | 95 | 0 | 0.679 | 0.711 | +0.032 |

## Do the two graders agree?

| system | n | exact | binary (usable vs not) |
|---|---|---|---|
| `closed_book` | 95 | 53.7% | 54.7% |
| `bm25` | 95 | 78.9% | 86.3% |
| `dense` | 95 | 77.9% | 87.4% |
| `hybrid` | 95 | 83.2% | 89.5% |
| `hybrid_rerank` | 95 | 78.9% | 81.1% |

## The claim that survives, and the one that does not

Best retrieval system by baseline accuracy: `bm25`.

- gain under `deepseek-chat [baseline]`: **+0.479** (0.221 -> 0.700)
- gain under `deepseek-reasoner [reasoner]`: **+0.353** (0.305 -> 0.658)

**Ranking stability**: changed under both graders.

- under `deepseek-chat [baseline]`: `closed_book < dense < hybrid_rerank < hybrid < bm25`
- under `deepseek-reasoner [reasoner]`: `closed_book < dense < bm25 < hybrid < hybrid_rerank`

`bm25`, `hybrid_rerank` change position. Before quoting that as a finding, check the gap: the largest difference among the movers is **0.021**, which at n≈95 is well inside one item. An ordering that flips under a grader swap and whose members are a fraction of an item apart is not a finding — report it as unresolved.

- closed-book is the worst system under both graders: **True**
- the best system is the same under both: **False** (`bm25`)

So the defensible claims are exactly those two, plus the gain interval. Any ordering between systems that swap places inside the noise band is unresolved, not a result.

## Verdict transitions

- `closed_book`: incorrect -> partial (29), partial -> incorrect (12), correct -> incorrect (1), incorrect -> correct (1), correct -> partial (1)
- `bm25`: correct -> partial (6), incorrect -> partial (6), partial -> incorrect (5), correct -> incorrect (2), partial -> correct (1)
- `dense`: correct -> partial (7), incorrect -> partial (6), partial -> incorrect (4), partial -> correct (2), correct -> incorrect (2)
- `hybrid`: incorrect -> partial (4), correct -> incorrect (3), correct -> partial (3), partial -> incorrect (3), partial -> correct (3)
- `hybrid_rerank`: incorrect -> partial (12), correct -> incorrect (3), partial -> incorrect (2), correct -> partial (1), partial -> correct (1), incorrect -> correct (1)
