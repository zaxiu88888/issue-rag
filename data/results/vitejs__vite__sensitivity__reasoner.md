# Grader sensitivity — Vite (`vitejs/vite`)

- baseline judge: `deepseek-chat [baseline]`
- variant judge: `deepseek-reasoner [reasoner]`
- answers are identical in both runs; only the grader differs (`evaluate --regrade`)

## Accuracy under each grader

| system | n | dropped | `deepseek-chat [baseline]` | `deepseek-reasoner [reasoner]` | delta |
|---|---|---|---|---|---|
| `closed_book` | 29 | 2 | 0.138 | 0.328 | +0.190 |
| `bm25` | 29 | 2 | 0.483 | 0.414 | -0.069 |
| `dense` | 29 | 0 | 0.448 | 0.362 | -0.086 |
| `hybrid` | 29 | 0 | 0.466 | 0.466 | +0.000 |
| `hybrid_rerank` | 29 | 0 | 0.586 | 0.483 | -0.103 |

_4 item(s) across all systems were dropped because one side lacked a usable verdict (a failed call, or a grader that emitted no parseable JSON). They are excluded rather than scored, because a grader failure is not a wrong answer._

## Do the two graders agree?

| system | n | exact | binary (usable vs not) |
|---|---|---|---|
| `closed_book` | 29 | 48.3% | 48.3% |
| `bm25` | 29 | 72.4% | 79.3% |
| `dense` | 29 | 62.1% | 65.5% |
| `hybrid` | 29 | 93.1% | 96.6% |
| `hybrid_rerank` | 29 | 86.2% | 89.7% |

## The claim that survives, and the one that does not

Best retrieval system by baseline accuracy: `hybrid_rerank`.

- gain under `deepseek-chat [baseline]`: **+0.448** (0.138 -> 0.586)
- gain under `deepseek-reasoner [reasoner]`: **+0.155** (0.328 -> 0.483)

**Ranking stability**: changed under both graders.

- under `deepseek-chat [baseline]`: `closed_book < dense < hybrid < bm25 < hybrid_rerank`
- under `deepseek-reasoner [reasoner]`: `closed_book < dense < bm25 < hybrid < hybrid_rerank`

`bm25`, `hybrid` change position. Before quoting that as a finding, check the gap: the largest difference among the movers is **0.017**, which at n≈29 is well inside one item. An ordering that flips under a grader swap and whose members are a fraction of an item apart is not a finding — report it as unresolved.

- closed-book is the worst system under both graders: **True**
- the best system is the same under both: **True** (`hybrid_rerank`)

So the defensible claims are exactly those two, plus the gain interval. Any ordering between systems that swap places inside the noise band is unresolved, not a result.

## Verdict transitions

- `closed_book`: incorrect -> partial (13), partial -> incorrect (2)
- `bm25`: partial -> incorrect (3), correct -> incorrect (2), partial -> correct (2), incorrect -> partial (1)
- `dense`: partial -> incorrect (4), incorrect -> partial (4), correct -> incorrect (2), correct -> partial (1)
- `hybrid`: incorrect -> partial (1), correct -> partial (1)
- `hybrid_rerank`: correct -> incorrect (2), correct -> partial (1), partial -> incorrect (1)
