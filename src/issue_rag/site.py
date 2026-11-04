"""把结果导出成单文件静态站点（可被 GitHub Pages 直接托管）。

Export a static, single-file results site.

Why this exists: the live demo needs a server and an API key, so a reader who
lands on the repository cannot try anything. `serve.py` is right for the author
and wrong for a stranger. This module writes one self-contained HTML file with no
JavaScript dependencies, no network calls and no build step, so it can live in
`docs/index.html` and be served by GitHub Pages straight from the repo.

It is a *report*, not a second implementation: every number it shows is read from
`data/results/`, which is produced by `issue_rag.evaluate`. Nothing is computed
here except percentages of already-stored verdicts, so the page cannot drift away
from the measured results.

Usage::

    python -m issue_rag.site --preset vite --out docs
"""

from __future__ import annotations

import argparse
import html
import json
import pathlib
from typing import Any, Iterable

from .config import RESULTS_DIR, RepoPreset, load_preset
from .corpus import load_jsonl

SYSTEM_LABELS = {
    "closed_book": "closed book (no retrieval)",
    "bm25": "BM25",
    "dense": "dense (bge-base-en-v1.5)",
    "hybrid": "hybrid (RRF)",
    "hybrid_rerank": "hybrid + rerank",
}


def _esc(text: Any) -> str:
    return html.escape(str(text if text is not None else ""))


def load_arm(preset: RepoPreset, system: str, tag: str = "") -> dict[str, dict[str, Any]]:
    """Last record per item for one (system, tag), excluding failed calls."""
    suffix = f"__{tag}" if tag else ""
    path = RESULTS_DIR / f"{preset.slug}__{system}{suffix}.jsonl"
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for record in load_jsonl(path):
        if record.get("error"):
            continue
        out[record["item_id"]] = record
    return out


def _summaries(preset: RepoPreset, tag: str = "") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    suffix = f"__{tag}__results" if tag else "__results"
    path = RESULTS_DIR / f"{preset.slug}{suffix}.json"
    if not path.exists():
        return [], {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("summaries", []), payload


def _pct(value: float | None) -> str:
    return f"{value:.1f}%" if value is not None else "—"


def _num(value: float | None, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if value is not None else "—"


def _table(headers: Iterable[str], rows: Iterable[Iterable[str]]) -> str:
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _accuracy_table(summaries: list[dict[str, Any]]) -> str:
    rows = []
    for row in summaries:
        name = SYSTEM_LABELS.get(row["system"], row["system"])
        rows.append(
            [
                f"<code>{_esc(name)}</code>",
                str(row.get("n", 0)),
                f"<b>{_num(row.get('accuracy'))}</b>",
                _pct(row.get("correct_rate")),
                _pct(row.get("partial_rate")),
                _pct(row.get("incorrect_rate")),
                _pct(row.get("fabricated_rate")),
                _num(row.get("latency_p50_s"), 2),
            ]
        )
    return _table(
        ["system", "n", "accuracy", "correct", "partial", "incorrect", "fabricated", "p50 s"],
        rows,
    )


def _retrieval_table(retrieval: dict[str, dict[str, float]]) -> str:
    rows = []
    for name, metrics in retrieval.items():
        rows.append(
            [
                f"<code>{_esc(name)}</code>",
                _num(metrics.get("gold_chunk@5")),
                _num(metrics.get("gold_chunk_mrr")),
                _num(metrics.get("gold_chunk_top1")),
                _num(metrics.get("self_share@5")),
            ]
        )
    return _table(
        ["retriever", "gold chunk@5", "gold chunk MRR", "gold chunk top-1", "window from same issue"],
        rows,
    )


def _rubric_ab(preset: RepoPreset) -> str:
    """Rubric ① vs ③, when both arms exist. Returns '' when the A/B is not run."""
    checks, _ = _summaries(preset)
    holistic, _ = _summaries(preset, "rubric1")
    if not checks or not holistic:
        return ""
    by_name = {row["system"]: row.get("accuracy") for row in holistic}
    rows = []
    for row in checks:
        alt = by_name.get(row["system"])
        if alt is None:
            continue
        rows.append(
            [
                f"<code>{_esc(SYSTEM_LABELS.get(row['system'], row['system']))}</code>",
                _num(alt),
                _num(row.get("accuracy")),
                f"{row.get('accuracy', 0) - alt:+.3f}",
            ]
        )
    if not rows:
        return ""
    return (
        "<h2>Rubric ① vs ③ (same answers, only the rubric differs)</h2>"
        "<p class='note'>Rubric ① asks for one verdict; rubric ③ asks four factual "
        "questions and maps them to a verdict in code. Both runs reuse the stored "
        "answers. The shipped instrument is ③ &mdash; ① exists only for this "
        "comparison.</p>"
        + _table(["system", "rubric ①", "rubric ③", "Δ"], rows)
    )


def _item_cards(preset: RepoPreset) -> tuple[str, int]:
    """One card per eval item, with both systems' answers side by side."""
    items = load_jsonl(preset.processed_dir / "evalset.jsonl")
    closed = load_arm(preset, "closed_book")
    rag = load_arm(preset, "hybrid_rerank")
    if not items or not closed or not rag:
        return "", 0

    cards = []
    for item in items:
        iid = item["id"]
        left = closed.get(iid, {})
        right = rag.get(iid, {})
        if not left and not right:
            continue
        verdicts = f"{left.get('verdict', '—')} / {right.get('verdict', '—')}"
        cards.append(
            f"""<article class="item" data-verdicts="{_esc(verdicts)}">
  <header>
    <span class="iid">{_esc(iid)}</span>
    <span class="q">{_esc(item.get('query'))}</span>
    <span class="tags"><i class="v{left.get('verdict', '')}">cb: {_esc(left.get('verdict', '—'))}</i>
      <i class="v{right.get('verdict', '')}">rag: {_esc(right.get('verdict', '—'))}</i></span>
  </header>
  <div class="body">
    <section><h4>reference (maintainer)</h4><pre>{_esc(item.get('reference_answer'))}</pre></section>
    <section><h4>closed book answer</h4><pre>{_esc(left.get('answer', '—'))}</pre></section>
    <section><h4>hybrid + rerank answer</h4><pre>{_esc(right.get('answer', '—'))}</pre></section>
  </div>
</article>"""
        )
    return "\n".join(cards), len(cards)


STYLE = """
:root{--ink:#16181d;--muted:#666e7a;--line:#e3e5e9;--bg:#fafbfc}
*{box-sizing:border-box}
body{margin:0 auto;padding:3rem 1.25rem 6rem;max-width:1000px;color:var(--ink);
  font:16px/1.6 -apple-system,"Segoe UI",Roboto,sans-serif;background:#fff}
h1{font-size:1.35rem;margin:0 0 .3rem}
h2{font-size:1rem;margin:2.6rem 0 .8rem;padding-bottom:.35rem;border-bottom:1px solid var(--line)}
h3{font-size:.86rem;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);
  margin:1.6rem 0 .5rem}
.lede{color:var(--muted);margin:0 0 .4rem;font-size:.94rem}
.note{color:var(--muted);font-size:.84rem}
table{width:100%;border-collapse:collapse;font-size:.88rem;margin:.4rem 0}
th,td{padding:.4rem .5rem;border-bottom:1px solid var(--line);text-align:left}
th{font-weight:600;color:var(--muted);font-size:.78rem;text-transform:uppercase;letter-spacing:.04em}
code{font-family:ui-monospace,Consolas,monospace;font-size:.9em;background:#f2f3f5;
  padding:.05rem .3rem;border-radius:3px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:.7rem;margin:1rem 0 0}
.card{border:1px solid var(--line);border-radius:8px;padding:.8rem .9rem;background:var(--bg)}
.card b{display:block;font-size:1.45rem;line-height:1.2}
.card span{color:var(--muted);font-size:.78rem}
.controls{display:flex;gap:.6rem;flex-wrap:wrap;align-items:center;margin:.8rem 0 1rem}
.controls input,.controls select{padding:.45rem .6rem;border:1px solid var(--line);
  border-radius:6px;font-size:.9rem}
.controls .count{color:var(--muted);font-size:.82rem;margin-left:auto}
.item{border:1px solid var(--line);border-radius:8px;margin:0 0 .55rem;overflow:hidden}
.item header{display:flex;gap:.6rem;align-items:baseline;padding:.55rem .8rem;cursor:pointer;
  background:var(--bg)}
.item .iid{font-family:ui-monospace,Consolas,monospace;font-size:.76rem;color:var(--muted)}
.item .q{font-weight:600;font-size:.9rem;flex:1}
.item .tags{display:flex;gap:.3rem}
.item i.v{font-style:normal;font-size:.7rem;padding:.05rem .35rem;border-radius:9px;
  background:#eceef1;color:var(--muted);white-space:nowrap}
.item i.vcorrect{background:#e3f6e8;color:#1a7f37}
.item i.vpartial{background:#fdf1d8;color:#9a6700}
.item i.vincorrect{background:#fdeaea;color:#b42318}
.item .body{display:none;padding:.2rem .8rem .8rem}
.item.open .body{display:block}
.item section{margin:.55rem 0 0}
.item h4{margin:0 0 .2rem;font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;
  color:var(--muted)}
.item pre{margin:0;white-space:pre-wrap;word-break:break-word;font-size:.82rem;
  font-family:inherit;background:#fff;border-left:2px solid var(--line);padding:.35rem .6rem}
footer{margin-top:3rem;color:var(--muted);font-size:.8rem;border-top:1px solid var(--line);
  padding-top:1rem}
"""

SCRIPT = """
(function () {
  const items = Array.from(document.querySelectorAll('.item'));
  const search = document.getElementById('search');
  const filter = document.getElementById('filter');
  const count = document.getElementById('count');

  function apply() {
    const needle = (search.value || '').toLowerCase();
    let shown = 0;
    items.forEach(function (el) {
      const text = el.textContent.toLowerCase();
      const v = el.dataset.verdicts;
      let ok = needle === '' || text.indexOf(needle) !== -1;
      if (ok && filter.value === 'closed-book-missed') ok = v.indexOf('correct') !== 0;
      if (ok && filter.value === 'both-right') ok = v.split(' / ').every(function (x) { return x === 'correct'; });
      if (ok && filter.value === 'disagree') {
        const parts = v.split(' / ');
        ok = parts[0] !== parts[1];
      }
      el.style.display = ok ? '' : 'none';
      if (ok) shown += 1;
    });
    count.textContent = shown + ' / ' + items.length + ' items';
  }

  items.forEach(function (el) {
    el.querySelector('header').addEventListener('click', function () {
      el.classList.toggle('open');
    });
  });
  search.addEventListener('input', apply);
  filter.addEventListener('change', apply);
  apply();
})();
"""


def build(preset: RepoPreset, out_dir: pathlib.Path) -> pathlib.Path:
    summaries, payload = _summaries(preset)
    retrieval = payload.get("retrieval", {})
    cards, n_cards = _item_cards(preset)

    closed = next((r for r in summaries if r["system"] == "closed_book"), None)
    best = max((r for r in summaries if r["system"] != "closed_book"), key=lambda r: r.get("accuracy", 0), default=None)
    gain = ""
    if closed and best and closed.get("accuracy"):
        gain = f"{best['accuracy'] / closed['accuracy']:.2f}&times;"

    headline = [
        ("closed book", _num(closed.get("accuracy")) if closed else "—", "accuracy, no retrieval"),
        ("best retrieval", _num(best.get("accuracy")) if best else "—", SYSTEM_LABELS.get(best["system"], "") if best else ""),
        ("gain", gain or "—", "best retrieval &divide; closed book"),
        ("eval items", str(payload.get("n_items", "—")), "hand-reviewed, mined from issues"),
        ("judge", "deepseek-chat", "same model as generator"),
    ]
    card_html = "".join(
        "<div class='card'><b>%s</b><span>%s</span></div>" % (_esc(value), _esc(caption))
        for _label, value, caption in headline
    )

    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>issue-rag &mdash; results</title>
<style>{STYLE}</style>
<h1>issue-rag &mdash; measured results</h1>
<p class="lede">A retrieval-augmented assistant over <code>{_esc(preset.repo)}</code>
issues, with a hand-reviewed eval set mined from maintainer answers. Every number
below is generated by <code>python -m issue_rag.evaluate</code>; this page only
renders <code>data/results/</code>.</p>
<div class="cards">{card_html}</div>

<h2>Answer quality</h2>
{_accuracy_table(summaries)}
<p class="note">accuracy = correct(1.0) + partial(0.5). fabricated = the judge
flagged an invented version, issue number, API or path.</p>

<h2>Retrieval quality</h2>
{_retrieval_table(retrieval)}
<p class="note">Scored on the maintainer's answering <b>comment</b>. The last
column is the diagnosis of the earlier issue-level metric: 98&ndash;99% of each
retrieved window is chunks of the gold issue itself, so "did we find the right
issue" was 1.000 for every retriever by construction.</p>

{_rubric_ab(preset)}

<h2>All {n_cards} eval items</h2>
<p class="note">Click a row to expand. Both systems answer the same question; the
reference is what the maintainer actually said.</p>
<div class="controls">
  <input id="search" type="search" placeholder="search questions, answers, issue ids">
  <select id="filter">
    <option value="all">all items</option>
    <option value="closed-book-missed">closed book got it wrong</option>
    <option value="disagree">the two systems disagree</option>
    <option value="both-right">both correct</option>
  </select>
  <span class="count" id="count"></span>
</div>
{cards}

<footer>
Generated by <code>python -m issue_rag.site --preset {_esc(preset.key)}</code>.
Corpus built from harvested issues; the generator and judge are
<code>deepseek-chat</code>. The point of this project is not the demo &mdash; it is
the audit trail behind these numbers, which lives in the repository and in
<code>data/results/</code>.
</footer>
<script>{SCRIPT}</script>
</html>
"""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "index.html"
    target.write_text(page, encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a static results site")
    parser.add_argument("--preset", default="vite")
    parser.add_argument(
        "--out",
        default="docs",
        help="output directory; the default puts index.html at the root of docs/ so "
        "GitHub Pages can serve the site with 'Source: main /docs' and no config",
    )
    args = parser.parse_args(argv)

    preset = load_preset(args.preset)
    target = build(preset, pathlib.Path(args.out))
    size_kb = target.stat().st_size / 1024
    print(f"wrote {target}  ({size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
