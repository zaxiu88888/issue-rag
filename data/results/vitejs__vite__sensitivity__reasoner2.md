# Grader sensitivity — Vite (`vitejs/vite`)

- baseline judge: `deepseek-reasoner [reasoner]`
- variant judge: `deepseek-reasoner [reasoner2]`
- answers are identical in both runs; only the grader differs (`evaluate --regrade`)

> **Both runs used the same model.** This table therefore measures the grader's own run-to-run noise, which sets the floor below which a difference between two systems must not be read as a result.

## Accuracy under each grader

| system | n | dropped | `deepseek-reasoner [reasoner]` | `deepseek-reasoner [reasoner2]` | delta |
|---|---|---|---|---|---|
| `closed_book` | 29 | 0 | 0.328 | 0.345 | +0.017 |
| `bm25` | 29 | 0 | 0.414 | 0.397 | -0.017 |
| `dense` | 28 | 1 | 0.339 | 0.375 | +0.036 |
| `hybrid` | 29 | 0 | 0.466 | 0.431 | -0.035 |
| `hybrid_rerank` | 29 | 0 | 0.483 | 0.552 | +0.069 |

_1 item(s) across all systems were dropped because one side lacked a usable verdict (a failed call, or a grader that emitted no parseable JSON). They are excluded rather than scored, because a grader failure is not a wrong answer._

## Do the two graders agree?

| system | n | exact | binary (usable vs not) |
|---|---|---|---|
| `closed_book` | 29 | 75.9% | 75.9% |
| `bm25` | 29 | 79.3% | 89.7% |
| `dense` | 28 | 71.4% | 75.0% |
| `hybrid` | 29 | 93.1% | 93.1% |
| `hybrid_rerank` | 29 | 82.8% | 86.2% |

## The claim that survives, and the one that does not

Best retrieval system by baseline accuracy: `hybrid_rerank`.

- gain under `deepseek-reasoner [reasoner]`: **+0.155** (0.328 -> 0.483)
- gain under `deepseek-reasoner [reasoner2]`: **+0.207** (0.345 -> 0.552)

**Ranking stability**: identical under both graders.

- under `deepseek-reasoner [reasoner]`: `closed_book < dense < bm25 < hybrid < hybrid_rerank`
- under `deepseek-reasoner [reasoner2]`: `closed_book < dense < bm25 < hybrid < hybrid_rerank`

- closed-book is the worst system under both graders: **True**
- the best system is the same under both: **True** (`hybrid_rerank`)

So the defensible claims are exactly those two, plus the gain interval. Any ordering between systems that swap places inside the noise band is unresolved, not a result.

## Verdict transitions

- `closed_book`: incorrect -> partial (4), partial -> incorrect (3)
- `bm25`: correct -> partial (3), incorrect -> correct (1), incorrect -> partial (1), partial -> incorrect (1)
- `dense`: incorrect -> partial (4), partial -> incorrect (3), partial -> correct (1)
- `hybrid`: partial -> incorrect (2)
- `hybrid_rerank`: incorrect -> partial (2), partial -> correct (1), partial -> incorrect (1), incorrect -> correct (1)
