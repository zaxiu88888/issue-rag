# Human vs judge — Vite

- reviewed items: **30**
- _3 duplicate decision line(s) collapsed to the latest per item_
- exact agreement: **17/30 = 56.7%**
- usable-vs-not agreement: **20/30 = 66.7%**
- quadratic-weighted kappa: **0.607** (substantial)

_This is the only measurement in the project that is not a model grading a
model. It is one reviewer over one sample: treat it as a validation of the
instrument's direction, not as a precise estimate._

| system | n | exact | usable | mean human | mean judge |
|---|---|---|---|---|---|
| `closed_book` | 15 | 47% | 53% | 0.100 | 0.267 |
| `hybrid_rerank` | 15 | 67% | 80% | 0.567 | 0.467 |

_`mean human` vs `mean judge` is the bias that matters: if the human scores
closed-book lower and retrieval higher, the judge is understating the gap._

## Disagreements (13)

- `hybrid_rerank:q18811` human **correct**, judge **partial**
- `closed_book:q8412` human **partial**, judge **incorrect**
- `hybrid_rerank:q20202` human **partial**, judge **incorrect**
- `closed_book:q22225` human **incorrect**, judge **partial**
- `closed_book:q21162` human **incorrect**, judge **partial**
- `closed_book:q21866` human **incorrect**, judge **partial**
- `closed_book:q18489` human **partial**, judge **correct**
- `hybrid_rerank:q20867` human **partial**, judge **correct**
- `hybrid_rerank:q22750` human **partial**, judge **incorrect**
- `closed_book:q21884` human **incorrect**, judge **partial**
- `closed_book:q20202` human **incorrect**, judge **correct**
- `hybrid_rerank:q22583` human **partial**, judge **incorrect**
- `closed_book:q22750` human **partial**, judge **incorrect**
