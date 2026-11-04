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

The page chrome (headings, table headers, notes, filters) ships in English and
Simplified Chinese behind a client-side toggle. Only the *chrome* is translated:
eval data — questions, maintainer references, generated answers — is rendered
verbatim, because it is the thing being measured and must not be re-written.

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

# Human-facing labels in the other language. Model names and identifiers stay
# untranslated on purpose; only the descriptive part moves.
SYSTEM_LABELS_ZH = {
    "closed_book": "闭卷（无检索）",
    "bm25": "BM25",
    "dense": "稠密检索 (bge-base-en-v1.5)",
    "hybrid": "混合检索 (RRF)",
    "hybrid_rerank": "混合 + 重排序",
}


def _esc(text: Any) -> str:
    return html.escape(str(text if text is not None else ""))


def _sys_code(system: str) -> str:
    """A system name rendered as a translatable <code> cell."""
    return f'<code data-i18n="sys.{_esc(system)}">{_esc(SYSTEM_LABELS.get(system, system))}</code>'


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


def _table(headers: Iterable[tuple[str, str]], rows: Iterable[Iterable[str]]) -> str:
    head = "".join(f'<th data-i18n="{_esc(key)}">{_esc(text)}</th>' for key, text in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _accuracy_table(summaries: list[dict[str, Any]]) -> str:
    rows = []
    for row in summaries:
        rows.append(
            [
                _sys_code(row["system"]),
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
        [
            ("th.system", "system"),
            ("th.n", "n"),
            ("th.accuracy", "accuracy"),
            ("th.correct", "correct"),
            ("th.partial", "partial"),
            ("th.incorrect", "incorrect"),
            ("th.fabricated", "fabricated"),
            ("th.p50", "p50 s"),
        ],
        rows,
    )


def _retrieval_table(retrieval: dict[str, dict[str, float]]) -> str:
    rows = []
    for name, metrics in retrieval.items():
        rows.append(
            [
                _sys_code(name),
                _num(metrics.get("gold_chunk@5")),
                _num(metrics.get("gold_chunk_mrr")),
                _num(metrics.get("gold_chunk_top1")),
                _num(metrics.get("self_share@5")),
            ]
        )
    return _table(
        [
            ("th.retriever", "retriever"),
            ("th.gold_chunk_at5", "gold chunk@5"),
            ("th.gold_chunk_mrr", "gold chunk MRR"),
            ("th.gold_chunk_top1", "gold chunk top-1"),
            ("th.self_share", "window from same issue"),
        ],
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
                _sys_code(row["system"]),
                _num(alt),
                _num(row.get("accuracy")),
                f"{row.get('accuracy', 0) - alt:+.3f}",
            ]
        )
    if not rows:
        return ""
    return (
        '<h2 data-i18n="sec.rubric">Rubric ① vs ③ (same answers, only the rubric differs)</h2>'
        '<p class="note" data-i18n="note.rubric">Rubric ① asks for one verdict; rubric ③ asks '
        "four factual questions and maps them to a verdict in code. Both runs reuse the stored "
        "answers. The shipped instrument is ③ — ① exists only for this comparison.</p>"
        + _table(
            [
                ("th.system", "system"),
                ("th.rubric1", "rubric ①"),
                ("th.rubric3", "rubric ③"),
                ("th.delta", "Δ"),
            ],
            rows,
        )
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
        lv = left.get("verdict", "—")
        rv = right.get("verdict", "—")
        verdicts = f"{lv} / {rv}"
        cards.append(
            f"""<article class="item" data-verdicts="{_esc(verdicts)}">
  <header>
    <span class="iid">{_esc(iid)}</span>
    <span class="q">{_esc(item.get('query'))}</span>
    <span class="tags"><i class="v{_esc(lv)}"><span data-i18n="tag.cb">cb</span>: <span data-i18n="verdict.{_esc(lv)}">{_esc(lv)}</span></i>
      <i class="v{_esc(rv)}"><span data-i18n="tag.rag">rag</span>: <span data-i18n="verdict.{_esc(rv)}">{_esc(rv)}</span></i></span>
  </header>
  <div class="body">
    <section><h4 data-i18n="item.reference">reference (maintainer)</h4><pre>{_esc(item.get('reference_answer'))}</pre></section>
    <section><h4 data-i18n="item.closed_book_answer">closed book answer</h4><pre>{_esc(left.get('answer', '—'))}</pre></section>
    <section><h4 data-i18n="item.rag_answer">hybrid + rerank answer</h4><pre>{_esc(right.get('answer', '—'))}</pre></section>
  </div>
</article>"""
        )
    return "\n".join(cards), len(cards)


def _translations(preset: RepoPreset, n_cards: int) -> dict[str, dict[str, str]]:
    """UI copy for the two languages, keyed by the `data-i18n` target.

    Values are `{en, zh}`. HTML inside values is intentional (innerHTML): only the
    page chrome carries it, and the measured data is never fed through here.
    """
    repo = _esc(preset.repo)
    t: dict[str, dict[str, str]] = {
        "title": {"en": "issue-rag — results", "zh": "issue-rag — 实测结果"},
        "h1": {"en": "issue-rag — measured results", "zh": "issue-rag — 实测结果"},
        "lede": {
            "en": (
                f"A retrieval-augmented assistant over <code>{repo}</code> issues, "
                "with a hand-reviewed eval set mined from maintainer answers. Every "
                "number below is generated by <code>python -m issue_rag.evaluate</code>; "
                "this page only renders <code>data/results/</code>."
            ),
            "zh": (
                f"面向 <code>{repo}</code> issue 的检索增强问答助手，评测集从维护者回答中"
                "人工复核挖掘而来。下面每个数字都由 <code>python -m issue_rag.evaluate</code> "
                "生成；本页只是渲染 <code>data/results/</code>。"
            ),
        },
        "card.closed_book": {"en": "accuracy, no retrieval", "zh": "准确率（无检索）"},
        "card.best_retrieval": {"en": "best retrieval", "zh": "最好检索"},
        "card.gain": {"en": "best retrieval ÷ closed book", "zh": "最好检索 ÷ 闭卷"},
        "card.eval_items": {"en": "hand-reviewed, mined from issues", "zh": "人工复核，从 issue 挖掘"},
        "card.judge": {"en": "same model as generator", "zh": "与生成器同一模型"},
        "sec.answer_quality": {"en": "Answer quality", "zh": "答案质量"},
        "note.accuracy": {
            "en": "accuracy = correct(1.0) + partial(0.5). fabricated = the judge flagged an invented version, issue number, API or path.",
            "zh": "accuracy = correct(1.0) + partial(0.5)。fabricated = 判分器标记出编造的版本号、issue 号、API 或路径。",
        },
        "sec.retrieval_quality": {"en": "Retrieval quality", "zh": "检索质量"},
        "note.retrieval": {
            "en": "Scored on the maintainer's answering <b>comment</b>. The last column is the diagnosis of the earlier issue-level metric: 98–99% of each retrieved window is chunks of the gold issue itself, so \"did we find the right issue\" was 1.000 for every retriever by construction.",
            "zh": "按维护者的<b>回答评论</b>打分。最后一列是对早前 issue 级指标的诊断：每个检索窗口里 98–99% 都是金标 issue 自身的 chunk，所以「有没有找到正确 issue」对每个检索器都按构造恒为 1.000。",
        },
        "sec.rubric": {"en": "Rubric ① vs ③ (same answers, only the rubric differs)", "zh": "Rubric ① vs ③（同一批答案，只换 rubric）"},
        "note.rubric": {
            "en": "Rubric ① asks for one verdict; rubric ③ asks four factual questions and maps them to a verdict in code. Both runs reuse the stored answers. The shipped instrument is ③ — ① exists only for this comparison.",
            "zh": "Rubric ① 直接要一个判定；rubric ③ 问四个事实性子问题，并由代码映射成判定。两次都复用已存答案。实际发布的是 ③ —— ① 只为这个对照存在。",
        },
        "sec.all_items": {"en": f"All {n_cards} eval items", "zh": f"全部 {n_cards} 条评测"},
        "note.items": {
            "en": "Click a row to expand. Both systems answer the same question; the reference is what the maintainer actually said.",
            "zh": "点击一行展开。两个系统回答同一问题；reference 是维护者真实所说。",
        },
        "search.placeholder": {"en": "search questions, answers, issue ids", "zh": "搜索问题、答案、issue 号"},
        "filter.all": {"en": "all items", "zh": "全部条目"},
        "filter.closed_book_missed": {"en": "closed book got it wrong", "zh": "闭卷答错"},
        "filter.disagree": {"en": "the two systems disagree", "zh": "两个系统不一致"},
        "filter.both_right": {"en": "both correct", "zh": "都正确"},
        "count": {"en": "{shown} / {total} items", "zh": "{shown} / {total} 条"},
        "item.reference": {"en": "reference (maintainer)", "zh": "参考答案（维护者）"},
        "item.closed_book_answer": {"en": "closed book answer", "zh": "闭卷答案"},
        "item.rag_answer": {"en": "hybrid + rerank answer", "zh": "混合 + 重排序答案"},
        "tag.cb": {"en": "cb", "zh": "闭卷"},
        "tag.rag": {"en": "rag", "zh": "检索"},
        "verdict.correct": {"en": "correct", "zh": "正确"},
        "verdict.partial": {"en": "partial", "zh": "部分"},
        "verdict.incorrect": {"en": "incorrect", "zh": "错误"},
        "toggle": {"en": "中文", "zh": "English"},
        "footer": {
            "en": (
                "Generated by <code>python -m issue_rag.site --preset "
                f"{_esc(preset.key)}</code>. Corpus built from harvested issues; the "
                "generator and judge are <code>deepseek-chat</code>. The point of this "
                "project is not the demo — it is the audit trail behind these numbers, "
                "which lives in the repository and in <code>data/results/</code>."
            ),
            "zh": (
                "由 <code>python -m issue_rag.site --preset "
                f"{_esc(preset.key)}</code> 生成。语料从抓取的 issue 构建；生成器与判分器是 "
                "<code>deepseek-chat</code>。这个项目的重点不是演示，而是这些数字背后的审计"
                "链条，它们在仓库和 <code>data/results/</code> 里。"
            ),
        },
        "th.system": {"en": "system", "zh": "系统"},
        "th.n": {"en": "n", "zh": "n"},
        "th.accuracy": {"en": "accuracy", "zh": "准确率"},
        "th.correct": {"en": "correct", "zh": "正确"},
        "th.partial": {"en": "partial", "zh": "部分"},
        "th.incorrect": {"en": "incorrect", "zh": "错误"},
        "th.fabricated": {"en": "fabricated", "zh": "编造"},
        "th.p50": {"en": "p50 s", "zh": "p50 秒"},
        "th.retriever": {"en": "retriever", "zh": "检索器"},
        "th.gold_chunk_at5": {"en": "gold chunk@5", "zh": "金标 chunk@5"},
        "th.gold_chunk_mrr": {"en": "gold chunk MRR", "zh": "金标 chunk MRR"},
        "th.gold_chunk_top1": {"en": "gold chunk top-1", "zh": "金标 chunk top-1"},
        "th.self_share": {"en": "window from same issue", "zh": "窗口里来自同一 issue 的比例"},
        "th.rubric1": {"en": "rubric ①", "zh": "rubric ①"},
        "th.rubric3": {"en": "rubric ③", "zh": "rubric ③"},
        "th.delta": {"en": "Δ", "zh": "Δ"},
    }
    for sys_key, en_label in SYSTEM_LABELS.items():
        t[f"sys.{sys_key}"] = {"en": en_label, "zh": SYSTEM_LABELS_ZH.get(sys_key, en_label)}
    return t


STYLE = """
:root{--ink:#16181d;--muted:#666e7a;--line:#e3e5e9;--bg:#fafbfc;--accent:#0b5cad}
*{box-sizing:border-box}
body{margin:0 auto;padding:3rem 1.25rem 6rem;max-width:1000px;color:var(--ink);
  font:16px/1.6 -apple-system,"Segoe UI",Roboto,sans-serif;background:#fff}
.lang{display:flex;justify-content:flex-end;margin:0 0 .6rem}
.lang button{padding:.3rem .7rem;border:1px solid var(--line);border-radius:6px;
  background:#fff;color:var(--ink);cursor:pointer;font-size:.85rem}
.lang button:hover{border-color:var(--accent);color:var(--accent)}
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
  const toggle = document.getElementById('lang-toggle');

  let lang = 'en';
  try { lang = localStorage.getItem('issue-rag-lang') || 'en'; } catch (e) {}

  function t(key) {
    const entry = I18N[key];
    if (!entry) return '';
    if (entry[lang] !== undefined) return entry[lang];
    return entry.en !== undefined ? entry.en : '';
  }

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
    const tpl = t('count') || '{shown} / {total} items';
    count.textContent = tpl.replace('{shown}', String(shown)).replace('{total}', String(items.length));
  }

  function applyLang() {
    document.documentElement.lang = (lang === 'zh') ? 'zh-CN' : 'en';
    document.title = t('title') || document.title;
    if (toggle) toggle.textContent = t('toggle');
    document.querySelectorAll('[data-i18n]').forEach(function (el) {
      const val = t(el.getAttribute('data-i18n'));
      if (val !== '') el.innerHTML = val;
    });
    document.querySelectorAll('[data-i18n-ph]').forEach(function (el) {
      const val = t(el.getAttribute('data-i18n-ph'));
      if (val !== '') el.setAttribute('placeholder', val);
    });
    apply();
  }

  items.forEach(function (el) {
    el.querySelector('header').addEventListener('click', function () {
      el.classList.toggle('open');
    });
  });
  search.addEventListener('input', apply);
  filter.addEventListener('change', apply);
  if (toggle) {
    toggle.addEventListener('click', function () {
      lang = (lang === 'en') ? 'zh' : 'en';
      try { localStorage.setItem('issue-rag-lang', lang); } catch (e) {}
      applyLang();
    });
  }
  applyLang();
})();
"""


def build(preset: RepoPreset, out_dir: pathlib.Path) -> pathlib.Path:
    summaries, payload = _summaries(preset)
    retrieval = payload.get("retrieval", {})
    cards, n_cards = _item_cards(preset)
    t = _translations(preset, n_cards)

    closed = next((r for r in summaries if r["system"] == "closed_book"), None)
    best = max(
        (r for r in summaries if r["system"] != "closed_book"),
        key=lambda r: r.get("accuracy", 0),
        default=None,
    )
    gain = ""
    if closed and best and closed.get("accuracy"):
        gain = f"{best['accuracy'] / closed['accuracy']:.2f}×"

    best_key = f"sys.{best['system']}" if best else "card.best_retrieval"
    headline = [
        (_num(closed.get("accuracy")) if closed else "—", "card.closed_book"),
        (_num(best.get("accuracy")) if best else "—", best_key),
        (gain or "—", "card.gain"),
        (str(payload.get("n_items", "—")), "card.eval_items"),
        ("deepseek-chat", "card.judge"),
    ]
    card_html = "".join(
        '<div class="card"><b>%s</b><span data-i18n="%s">%s</span></div>'
        % (_esc(value), _esc(key), t[key]["en"])
        for value, key in headline
    )

    i18n_json = json.dumps(t, ensure_ascii=True)

    page = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>%s</title>
<style>%s</style>
<div class="lang"><button id="lang-toggle" type="button" data-i18n="toggle">%s</button></div>
<h1 data-i18n="h1">%s</h1>
<p class="lede" data-i18n="lede">%s</p>
<div class="cards">%s</div>

<h2 data-i18n="sec.answer_quality">%s</h2>
%s
<p class="note" data-i18n="note.accuracy">%s</p>

<h2 data-i18n="sec.retrieval_quality">%s</h2>
%s
<p class="note" data-i18n="note.retrieval">%s</p>

%s

<h2 data-i18n="sec.all_items">%s</h2>
<p class="note" data-i18n="note.items">%s</p>
<div class="controls">
  <input id="search" type="search" data-i18n-ph="search.placeholder" placeholder="%s">
  <select id="filter">
    <option value="all" data-i18n="filter.all">%s</option>
    <option value="closed-book-missed" data-i18n="filter.closed_book_missed">%s</option>
    <option value="disagree" data-i18n="filter.disagree">%s</option>
    <option value="both-right" data-i18n="filter.both_right">%s</option>
  </select>
  <span class="count" id="count"></span>
</div>
%s

<footer data-i18n="footer">%s</footer>
<script>
const I18N = %s;
%s
</script>
</html>
""" % (
        t["title"]["en"],
        STYLE,
        t["toggle"]["en"],
        t["h1"]["en"],
        t["lede"]["en"],
        card_html,
        t["sec.answer_quality"]["en"],
        _accuracy_table(summaries),
        t["note.accuracy"]["en"],
        t["sec.retrieval_quality"]["en"],
        _retrieval_table(retrieval),
        t["note.retrieval"]["en"],
        _rubric_ab(preset),
        t["sec.all_items"]["en"],
        t["note.items"]["en"],
        t["search.placeholder"]["en"],
        t["filter.all"]["en"],
        t["filter.closed_book_missed"]["en"],
        t["filter.disagree"]["en"],
        t["filter.both_right"]["en"],
        cards,
        t["footer"]["en"],
        i18n_json,
        SCRIPT,
    )
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
