# Grader sensitivity — Vite (`vitejs/vite`)

- baseline judge: `deepseek-chat [v1holistic]`
- variant judge: `deepseek-chat []`
- answers are identical in both runs; only the grader differs (`evaluate --regrade`)

> **Both runs used the same model.** This table therefore measures the grader's own run-to-run noise, which sets the floor below which a difference between two systems must not be read as a result.

## Accuracy under each grader

| system | n | dropped | `deepseek-chat [v1holistic]` | `deepseek-chat []` | delta |
|---|---|---|---|---|---|
| `closed_book` | 29 | 2 | 0.138 | 0.190 | +0.052 |
| `bm25` | 29 | 2 | 0.483 | 0.517 | +0.035 |
| `dense` | 29 | 0 | 0.448 | 0.466 | +0.017 |
| `hybrid` | 29 | 0 | 0.466 | 0.500 | +0.035 |
| `hybrid_rerank` | 29 | 0 | 0.586 | 0.500 | -0.086 |

_4 item(s) across all systems were dropped because one side lacked a usable verdict (a failed call, or a grader that emitted no parseable JSON). They are excluded rather than scored, because a grader failure is not a wrong answer._

## Do the two graders agree?

| system | n | exact | binary (usable vs not) |
|---|---|---|---|
| `closed_book` | 29 | 75.9% | 82.8% |
| `bm25` | 29 | 72.4% | 86.2% |
| `dense` | 29 | 65.5% | 65.5% |
| `hybrid` | 29 | 65.5% | 72.4% |
| `hybrid_rerank` | 29 | 82.8% | 82.8% |

## The claim that survives, and the one that does not

Best retrieval system by baseline accuracy: `hybrid_rerank`.

- gain under `deepseek-chat [v1holistic]`: **+0.448** (0.138 -> 0.586)
- gain under `deepseek-chat []`: **+0.310** (0.190 -> 0.500)

**Ranking stability**: changed under both graders.

- under `deepseek-chat [v1holistic]`: `closed_book < dense < hybrid < bm25 < hybrid_rerank`
- under `deepseek-chat []`: `closed_book < dense < hybrid < hybrid_rerank < bm25`

`bm25`, `hybrid_rerank` change position. Before quoting that as a finding, check the gap: the largest difference among the movers is **0.103**, which at n≈29 is well inside one item. An ordering that flips under a grader swap and whose members are a fraction of an item apart is not a finding — report it as unresolved.

- closed-book is the worst system under both graders: **True**
- the best system is the same under both: **False** (`hybrid_rerank`)

So the defensible claims are exactly those two, plus the gain interval. Any ordering between systems that swap places inside the noise band is unresolved, not a result.

## Verdict transitions

- `closed_book`: incorrect -> partial (3), partial -> incorrect (2), partial -> correct (2)
- `bm25`: partial -> correct (3), incorrect -> partial (2), partial -> incorrect (2), correct -> partial (1)
- `dense`: incorrect -> partial (6), partial -> incorrect (3), correct -> incorrect (1)
- `hybrid`: partial -> incorrect (4), incorrect -> partial (4), partial -> correct (2)
- `hybrid_rerank`: partial -> incorrect (5)
