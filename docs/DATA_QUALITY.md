# Data quality findings

Everything here was measured on real API responses during development. These are
the things that silently turn a promising project into an empty dataset, and
they are worth more in an interview than the final accuracy number, because
almost nobody reports them.

All figures come from the harvests described below; raw responses are cached
under `data/cache` and the derived files under `data/raw`.

---

## 1. Bot dominance can wipe out a repository

`vercel/ai`, 2000 harvested comments, 12-month window:

| author association | count |
|---|---|
| `CONTRIBUTOR` (mostly bots) | 1755 |
| `NONE` | 169 |
| `COLLABORATOR` | 57 |
| `MEMBER` | 19 |

Two accounts, `ai-sdk-factory[bot]` (1217 comments) and
`github-actions[bot]` (457), produced **86% of the sample**. After dropping
automated authors, 600 issues yielded:

- 63 usable comments
- 37 issues with any answer at all
- **10** eval candidates

An eval set of 10 is not an experiment. Note also that these bots carry
`CONTRIBUTOR` association, not `NONE` — filtering on association alone would
have missed them, so the bot filter has to be name-based (`[bot]`, `*-bot`,
`dependabot`, …) and applied before any statistics are computed.

**The one-request check that catches this.** Before committing to a target
repository, look at the 50 most recent comments and count maintainer association
and bot names. Measured for four candidates:

| repository | maintainer rate | bot rate | verdict |
|---|---|---|---|
| `vitejs/vite` | 40% | 20% | chosen |
| `ollama/ollama` | 32% | 0% | good alternative |
| `vercel/ai` | 8% | 84% | rejected |
| `langchain-ai/langchainjs` | 2% | 72% | rejected |

---

## 2. Sampling on two different clocks produces two unrelated samples

First attempt harvested issues with `created:>=<date>` and comments from the
repo-wide comment feed, which GitHub returns **sorted by update time**. The two
selections therefore describe different populations:

> 150 harvested comments → **12** documents survived the join.

The fix is to select issues on the same axis the comment feed is ordered by:
query `updated:>=<date>` instead of `created:>=<date>`. This is recorded in
`fetch.py --window-axis`, whose default is `updated`.

The general lesson: when joining two independently paginated API streams, join
keys are not enough — the *ordering and filtering predicates* have to match, or
the intersection is arbitrary.

---

## 3. The repo-wide comment feed is mostly pull-request comments

`GET /repos/{owner}/{repo}/issues/comments` returns comments on issues **and**
pull requests, because in GitHub's data model a PR is an issue. No parameter
filters them apart.

On `vitejs/vite`, 150 repo-wide comments spanned 96 distinct numbers, but only
**10** of those numbers appeared in the issue-only harvest:

- remaining 86 threads were pull requests, already excluded from the issue set,
  or issues outside the selected window.

The repo-wide endpoint is 100× cheaper per comment, but if most of those comments
are about PRs it is 100× cheaper at producing nothing. `fetch.py --mode per-issue`
trades one request per issue for a stream that is pure signal.

---

## 4. Issue numbers are shared between issues and PRs

A number like `#23889` identifies an issue *or* a pull request in the same
namespace. Any join keyed on the number alone will silently merge the two unless
privacy is handled explicitly, which is why the harvester records
`is_pull_request` per issue and the corpus builder drops those records rather
than relying on the query being correct.

---

## 5. Chunking turns 533 issues into ~1900 documents

`vitejs/vite`, 533 issues: 1930 body chunks, median 1178 characters, p95 1344,
max 1481. Bodies are chunked at 1200 characters on paragraph boundaries with 150
characters of overlap.

That ratio (~3.6 chunks per issue) is the number to watch: if it creeps up,
chunks are getting small enough to lose the subject, which is why every chunk is
prefixed with its issue title.

---

## What this means for the numbers

The eval set is built from maintainer answers only, so its size is bounded by
maintainer density × harvest coverage. A 60-request hourly quota covers roughly
60 issues per hour in `--mode per-issue`, so a 100-item eval set needs a couple
of hours of harvesting (or a token, which raises the quota to 5000/hour).

Report the eval-set size next to every accuracy number. A 95% accuracy on 12
items is not a result.
