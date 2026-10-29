# Grader sensitivity — Vite (`vitejs/vite`)

- baseline judge: `deepseek-chat [baseline]`
- variant judge: `deepseek-reasoner [v4a]`
- answers are identical in both runs; only the grader differs (`evaluate --regrade`)

## Accuracy under each grader

| system | n | dropped | `deepseek-chat [baseline]` | `deepseek-reasoner [v4a]` | delta |
|---|---|---|---|---|---|
| `closed_book` | 28 | 1 | 0.196 | 0.286 | +0.089 |
| `hybrid_rerank` | 28 | 1 | 0.518 | 0.554 | +0.036 |

_2 item(s) across all systems were dropped because one side lacked a usable verdict (a failed call, or a grader that emitted no parseable JSON). They are excluded rather than scored, because a grader failure is not a wrong answer._

## Do the two graders agree?

| system | n | exact | binary (usable vs not) |
|---|---|---|---|
| `closed_book` | 28 | 53.6% | 57.1% |
| `hybrid_rerank` | 28 | 75.0% | 75.0% |

## The claim that survives, and the one that does not

Best retrieval system by baseline accuracy: `hybrid_rerank`.

- gain under `deepseek-chat [baseline]`: **+0.322** (0.196 -> 0.518)
- gain under `deepseek-reasoner [v4a]`: **+0.268** (0.286 -> 0.554)

**Ranking stability**: identical under both graders.

- under `deepseek-chat [baseline]`: `closed_book < hybrid_rerank`
- under `deepseek-reasoner [v4a]`: `closed_book < hybrid_rerank`

- closed-book is the worst system under both graders: **True**
- the best system is the same under both: **True** (`hybrid_rerank`)

So the defensible claims are exactly those two, plus the gain interval. Any ordering between systems that swap places inside the noise band is unresolved, not a result.

## Verdict transitions

- `closed_book`: incorrect -> partial (8), partial -> incorrect (2), incorrect -> correct (1), correct -> partial (1), correct -> incorrect (1)
- `hybrid_rerank`: incorrect -> partial (5), partial -> incorrect (1), correct -> incorrect (1)
