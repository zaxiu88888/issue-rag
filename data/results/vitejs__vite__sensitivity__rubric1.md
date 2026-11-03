# Grader sensitivity — Vite (`vitejs/vite`)

- baseline judge: `deepseek-chat [baseline]`
- variant judge: `deepseek-chat [rubric1]`
- answers are identical in both runs; only the grader differs (`evaluate --regrade`)

> **Both runs used the same model.** This table therefore measures the grader's own run-to-run noise, which sets the floor below which a difference between two systems must not be read as a result.

## Accuracy under each grader

| system | n | dropped | `deepseek-chat [baseline]` | `deepseek-chat [rubric1]` | delta |
|---|---|---|---|---|---|
| `closed_book` | 95 | 0 | 0.221 | 0.174 | -0.047 |
| `bm25` | 95 | 0 | 0.700 | 0.737 | +0.037 |
| `dense` | 95 | 0 | 0.632 | 0.695 | +0.063 |
| `hybrid` | 95 | 0 | 0.695 | 0.716 | +0.021 |
| `hybrid_rerank` | 95 | 0 | 0.679 | 0.768 | +0.089 |

## Do the two graders agree?

| system | n | exact | binary (usable vs not) |
|---|---|---|---|
| `closed_book` | 95 | 62.1% | 65.3% |
| `bm25` | 95 | 84.2% | 90.5% |
| `dense` | 95 | 81.1% | 86.3% |
| `hybrid` | 95 | 81.1% | 92.6% |
| `hybrid_rerank` | 95 | 81.1% | 84.2% |

## The claim that survives, and the one that does not

Best retrieval system by baseline accuracy: `bm25`.

- gain under `deepseek-chat [baseline]`: **+0.479** (0.221 -> 0.700)
- gain under `deepseek-chat [rubric1]`: **+0.563** (0.174 -> 0.737)

**Ranking stability**: changed under both graders.

- under `deepseek-chat [baseline]`: `closed_book < dense < hybrid_rerank < hybrid < bm25`
- under `deepseek-chat [rubric1]`: `closed_book < dense < hybrid < bm25 < hybrid_rerank`

`bm25`, `hybrid`, `hybrid_rerank` change position. Before quoting that as a finding, check the gap: the largest difference among the movers is **0.021**, which at n≈95 is well inside one item. An ordering that flips under a grader swap and whose members are a fraction of an item apart is not a finding — report it as unresolved.

- closed-book is the worst system under both graders: **True**
- the best system is the same under both: **False** (`bm25`)

So the defensible claims are exactly those two, plus the gain interval. Any ordering between systems that swap places inside the noise band is unresolved, not a result.

## Verdict transitions

- `closed_book`: partial -> incorrect (20), incorrect -> partial (12), correct -> partial (3), incorrect -> correct (1)
- `bm25`: incorrect -> partial (7), partial -> correct (4), correct -> partial (2), partial -> incorrect (2)
- `dense`: incorrect -> partial (11), partial -> correct (4), partial -> incorrect (2), correct -> partial (1)
- `hybrid`: correct -> partial (6), incorrect -> partial (6), partial -> correct (5), partial -> incorrect (1)
- `hybrid_rerank`: incorrect -> partial (10), partial -> correct (3), incorrect -> correct (3), partial -> incorrect (2)
