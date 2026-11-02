# Human vs judge — Vite

- reviewed items: **30**
- exact agreement: **21/30 = 70.0%**
- usable-vs-not agreement: **22/30 = 73.3%**
- quadratic-weighted kappa: **0.816** (almost perfect)

_This is the only measurement in the project that is not a model grading a
model. It is one reviewer over one sample: treat it as a validation of the
instrument's direction, not as a precise estimate._

| system | n | exact | usable | mean human | mean judge |
|---|---|---|---|---|---|
| `closed_book` | 15 | 67% | 73% | 0.167 | 0.267 |
| `hybrid_rerank` | 15 | 73% | 73% | 0.867 | 0.733 |

_`mean human` vs `mean judge` is the bias that matters: if the human scores
closed-book lower and retrieval higher, the judge is understating the gap._

## Disagreements (9)

- `hybrid_rerank:q22225` human **partial**, judge **incorrect**
- `closed_book:q20721` human **incorrect**, judge **partial**
- `hybrid_rerank:q21889` human **partial**, judge **incorrect**
- `closed_book:q21876` human **incorrect**, judge **partial**
- `closed_book:q21139` human **partial**, judge **incorrect**
- `hybrid_rerank:q22009` human **partial**, judge **incorrect**
- `hybrid_rerank:q22750` human **partial**, judge **incorrect**
- `closed_book:q23430` human **partial**, judge **correct**
- `closed_book:q22009` human **incorrect**, judge **partial**
