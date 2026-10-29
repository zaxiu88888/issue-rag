"""Blind human review of a sample of graded answers.

Why this module exists
----------------------
Every number in this project comes from a model grading another model. The
cross-model check measures how *reproducible* that is; only a person can measure
whether it is *right*. Thirty items, graded blind, is the cheapest real validation
available and it cannot be automated.

Three things make it honest:

* **Blind.** The queue file stores the judge's verdict (needed for scoring) but the
  prompt never prints it, and it does not print which system produced the answer
  either — knowing an answer came from `closed_book` biases a reviewer toward being
  harsh.
* **Sampled independently per system.** A question can therefore appear twice with
  two different answers. That is intentional: it makes the per-system comparison
  paired, at the cost of re-reading the reference on the duplicate.
* **Resumable.** Decisions append to a JSONL keyed by item, so a session can be
  stopped and continued without re-reading anything.

Usage::

    # 1. build a queue (deterministic, seeded; 15 items from each of two systems)
    python -m issue_rag.human_review --preset vite --n 30

    # 2. grade it, one item at a time
    python -m issue_rag.human_review --preset vite --review

    # 3. compare against the judge
    python -m issue_rag.human_review --preset vite --score

Grade `correct` when the candidate states the same resolution as the reference
(wording may differ). `partial` when it is on topic and partly right but misses the
key step. `incorrect` when it contradicts, invents a version, or is only generic
advice that would apply to any project.
"""

from __future__ import annotations

import argparse
import json
import random
import textwrap
from pathlib import Path
from typing import Any

from .config import RepoPreset, load_preset
from .corpus import load_jsonl
from .evaluate import load_shard, shard_path

QUEUE_FILE = "human_review_queue.jsonl"
DECISIONS_FILE = "human_review.jsonl"
VERDICTS = ("correct", "partial", "incorrect")
SCORES = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
USABLE = {"correct", "partial"}

# Two systems, chosen because they are the extremes: closed_book is where graders
# disagree most, and the reranked system is where they agree most. Reviewing only
# one of them would report agreement from the easiest or the hardest end alone.
DEFAULT_SYSTEMS = "closed_book,hybrid_rerank"


def build_queue(
    preset: RepoPreset,
    *,
    systems: list[str],
    total: int,
    seed: int = 20261029,
) -> list[dict[str, Any]]:
    """Sample graded pairs to review, balanced across systems and deterministic."""
    items = {item["id"]: item for item in load_jsonl(preset.processed_dir / "evalset.jsonl")}
    if not items:
        raise SystemExit("no eval set; run issue_rag.evalset first")

    per_system = max(1, total // len(systems))
    rng = random.Random(seed)
    queue: list[dict[str, Any]] = []

    for system in systems:
        records = [
            rec
            for iid, rec in load_shard(shard_path(preset, system)).items()
            if iid in items and rec.get("verdict") in VERDICTS
        ]
        records.sort(key=lambda rec: rec["item_id"])
        chosen = rng.sample(records, min(per_system, len(records)))
        chosen.sort(key=lambda rec: rec["item_id"])
        for record in chosen:
            item = items[record["item_id"]]
            queue.append(
                {
                    "review_id": f"{system}:{record['item_id']}",
                    "item_id": record["item_id"],
                    "system": system,
                    "question": item["query"],
                    "reference": item["reference_answer"],
                    "candidate": record["answer"],
                    # Kept for scoring only. The prompt must never print these.
                    "judge_verdict": record["verdict"],
                    "judge_checks": record.get("checks") or {},
                }
            )

    rng.shuffle(queue)
    return queue


def load_decisions(path: Path) -> dict[str, dict[str, Any]]:
    return {rec["review_id"]: rec for rec in load_jsonl(path)}


def _wrap(text: str, width: int, indent: str = "    ") -> str:
    out = []
    for paragraph in (text or "").split("\n"):
        if not paragraph.strip():
            out.append("")
            continue
        out.extend(
            textwrap.wrap(
                paragraph, width=width, initial_indent=indent, subsequent_indent=indent
            )
            or [indent]
        )
    return "\n".join(out)


def render(entry: dict[str, Any], position: int, total: int, width: int) -> str:
    """One item, with the judge's verdict and the system name deliberately absent."""
    lines = [
        "",
        f"[{position}/{total}]   {entry['item_id']}",
        "-" * width,
        "QUESTION",
        _wrap(entry["question"], width),
        "",
        "REFERENCE (the maintainer's resolution)",
        _wrap(entry["reference"], width),
        "",
        "CANDIDATE (what the system answered)",
        _wrap(entry["candidate"], width),
        "-" * width,
    ]
    return "\n".join(lines)


def review(preset: RepoPreset, queue_path: Path, decisions_path: Path, width: int) -> int:
    queue = load_jsonl(queue_path)
    if not queue:
        raise SystemExit(f"no queue at {queue_path}; run --n 30 first")
    done = load_decisions(decisions_path)
    remaining = [entry for entry in queue if entry["review_id"] not in done]

    print(f"queue: {len(queue)} items | already graded: {len(done)} | remaining: {len(remaining)}")
    if not remaining:
        print("nothing left. Run --score.")
        return 0
    print(
        "Grade the CANDIDATE against the REFERENCE.\n"
        "  c = correct   (same resolution; wording may differ)\n"
        "  p = partial   (on topic, partly right, misses the key step)\n"
        "  i = incorrect (contradicts, invents a version, or only generic advice)\n"
        "  s = skip   q = quit and save\n"
    )

    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    with decisions_path.open("a", encoding="utf-8") as handle:
        for position, entry in enumerate(queue, start=1):
            if entry["review_id"] in done:
                continue
            print(render(entry, position, len(queue), width))
            while True:
                answer = input("verdict? [c/p/i/s/q] ").strip().lower()
                if answer in {"q", "quit"}:
                    print("saved; resume with --review")
                    return 0
                if answer in {"s", ""}:
                    break
                mapping = {"c": "correct", "p": "partial", "i": "incorrect"}
                if answer in mapping:
                    note = input("note (optional): ").strip()
                    handle.write(
                        json.dumps(
                            {
                                "review_id": entry["review_id"],
                                "item_id": entry["item_id"],
                                "system": entry["system"],
                                "human_verdict": mapping[answer],
                                "human_note": note,
                                "question": entry["question"],
                                "candidate": entry["candidate"],
                                "judge_verdict": entry["judge_verdict"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    handle.flush()
                    break
                print("  enter c, p, i, s or q")
    print("\nqueue complete. Run --score.")
    return 0


def quadratically_weighted_kappa(
    human: list[str], judge: list[str]
) -> float | None:
    """Agreement above chance for ordinal labels.

    Correct/partial/incorrect is an ordered scale, so a plain agreement rate
    understates how wrong a one-step disagreement is and overstates how wrong a
    two-step one is. Quadratic weights are the standard treatment for ordinal
    ratings; the result is comparable to published kappa values.
    """
    n = len(human)
    if n == 0:
        return None
    index = {name: i for i, name in enumerate(VERDICTS)}
    k = len(VERDICTS)
    observed = [[0.0] * k for _ in range(k)]
    for h, j in zip(human, judge):
        observed[index[h]][index[j]] += 1
    total = float(n)
    observed = [[cell / total for cell in row] for row in observed]
    human_marginal = [sum(observed[i]) for i in range(k)]
    judge_marginal = [sum(observed[i][j] for i in range(k)) for j in range(k)]

    def weight(a: int, b: int) -> float:
        return 1.0 - ((a - b) ** 2) / ((k - 1) ** 2)

    observed_agreement = sum(
        weight(i, j) * observed[i][j] for i in range(k) for j in range(k)
    )
    expected_agreement = sum(
        weight(i, j) * human_marginal[i] * judge_marginal[j]
        for i in range(k)
        for j in range(k)
    )
    if expected_agreement >= 1.0:
        return None
    return (observed_agreement - expected_agreement) / (1.0 - expected_agreement)


def score(preset: RepoPreset, decisions_path: Path) -> int:
    decisions = load_jsonl(decisions_path)
    if not decisions:
        raise SystemExit(f"no decisions at {decisions_path}; run --review first")

    human = [rec["human_verdict"] for rec in decisions]
    judge = [rec["judge_verdict"] for rec in decisions]
    n = len(human)
    exact = sum(1 for h, j in zip(human, judge) if h == j)
    binary = sum(1 for h, j in zip(human, judge) if (h in USABLE) == (j in USABLE))
    kappa = quadratically_weighted_kappa(human, judge)

    print(f"# Human vs judge — {preset.display_name}\n")
    print(f"- reviewed items: **{n}**")
    print(f"- exact agreement: **{exact}/{n} = {100.0 * exact / n:.1f}%**")
    print(f"- usable-vs-not agreement: **{binary}/{n} = {100.0 * binary / n:.1f}%**")
    if kappa is not None:
        interpretation = (
            "poor" if kappa < 0.20 else
            "fair" if kappa < 0.40 else
            "moderate" if kappa < 0.60 else
            "substantial" if kappa < 0.80 else
            "almost perfect"
        )
        print(f"- quadratic-weighted kappa: **{kappa:.3f}** ({interpretation})")
    print()

    per_system: dict[str, list[tuple[str, str]]] = {}
    for rec in decisions:
        per_system.setdefault(rec["system"], []).append(
            (rec["human_verdict"], rec["judge_verdict"])
        )
    print("| system | n | exact | usable | mean human | mean judge |")
    print("|---|---|---|---|---|---|")
    for system, pairs in sorted(per_system.items()):
        count = len(pairs)
        system_exact = sum(1 for h, j in pairs if h == j)
        system_usable = sum(1 for h, j in pairs if (h in USABLE) == (j in USABLE))
        print(
            f"| `{system}` | {count} | {100.0 * system_exact / count:.0f}% | "
            f"{100.0 * system_usable / count:.0f}% | "
            f"{sum(SCORES[h] for h, _ in pairs) / count:.3f} | "
            f"{sum(SCORES[j] for _, j in pairs) / count:.3f} |"
        )

    disagreements = [
        rec for rec in decisions if rec["human_verdict"] != rec["judge_verdict"]
    ]
    print(f"\n## Disagreements ({len(disagreements)})\n")
    for rec in disagreements:
        note = f" — {rec['human_note']}" if rec.get("human_note") else ""
        print(
            f"- `{rec['item_id']}` human **{rec['human_verdict']}**, "
            f"judge **{rec['judge_verdict']}**{note}"
        )
    if not disagreements:
        print("- none")

    print(
        "\nPaste the three summary lines into README's Limitations, and keep this "
        "file committed: it is the only human-graded evidence the project has."
    )
    return 0


PAGE = """<!doctype html>
<meta charset="utf-8">
<title>issue-rag 盲评</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ font: 15px/1.65 -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
         max-width: 900px; margin: 2rem auto; padding: 0 1.25rem; color: #1a1a1a; }}
  h1 {{ font-size: 1.15rem; margin: 0 0 .25rem; }}
  .bar {{ height: 6px; background: #e6e6e6; border-radius: 3px; overflow: hidden;
         margin: .75rem 0 1.5rem; }}
  .bar > i {{ display: block; height: 100%; background: #1a1a1a; width: 0; }}
  .meta {{ color: #666; font-size: .85rem; }}
  .block {{ border: 1px solid #e2e2e2; border-radius: 8px; padding: .85rem 1rem;
           margin: .85rem 0; white-space: pre-wrap; word-break: break-word; }}
  .label {{ font-size: .72rem; letter-spacing: .06em; text-transform: uppercase;
           color: #777; margin-bottom: .35rem; font-weight: 600; }}
  .q {{ background: #f7f8fa; }}
  .ref {{ background: #f2f7f2; border-color: #d8e6d8; }}
  .cand {{ background: #fbf7f0; border-color: #eae0cf; }}
  .row {{ display: flex; gap: .5rem; flex-wrap: wrap; margin-top: 1.25rem; }}
  button {{ font: inherit; padding: .6rem 1.1rem; border-radius: 7px; border: 1px solid #ccc;
           background: #fff; cursor: pointer; }}
  button:hover {{ background: #f2f2f2; }}
  button.primary {{ background: #1a1a1a; color: #fff; border-color: #1a1a1a; }}
  button.c {{ border-color: #2e7d32; color: #2e7d32; }}
  button.p {{ border-color: #b26a00; color: #b26a00; }}
  button.i {{ border-color: #b3261e; color: #b3261e; }}
  textarea {{ width: 100%; min-height: 42px; font: inherit; margin-top: .75rem;
            padding: .5rem; border: 1px solid #ddd; border-radius: 6px; }}
  .rules {{ font-size: .85rem; color: #444; background: #fafafa; border-radius: 8px;
           padding: .75rem 1rem; margin-bottom: 1rem; }}
  .rules b {{ color: #111; }}
  .done {{ font-size: 1.1rem; padding: 2rem 0; }}
  kbd {{ background:#eee; border:1px solid #ccc; border-radius:4px; padding:0 .35em;
        font: .85em ui-monospace, monospace; }}
  .warn {{ color:#b3261e; font-size:.85rem; min-height:1.2em; }}
</style>

<h1>issue-rag 盲评</h1>
<div class="meta">共 <b id="total">{total}</b> 条 · 已完成 <b id="done">0</b> 条 ·
  第 <b id="idx">1</b> 条</div>
<div class="bar"><i id="progress"></i></div>

<div class="rules">
  <b>c 正确</b>：说出的结论与参考答案一致（措辞可不同）。<br>
  <b>p 部分正确</b>：方向对但缺关键步骤，或说不到具体点。<br>
  <b>i 错误</b>：与参考矛盾、编造版本号/issue 号，或只是对任何项目都成立的泛泛建议。
  <br><br>
  两个容易判错的边界：泛泛建议（清缓存、升级 Node、给个复现）判 <b>i</b> 不是 p；
  修法对但附带编造的版本号判 <b>p</b> 不是 c。<br>
  快捷键：<kbd>c</kbd> <kbd>p</kbd> <kbd>i</kbd> 判定 · <kbd>s</kbd> 跳过 · 刷新页面不丢进度。
</div>

<div id="card"></div>
<div class="warn" id="warn"></div>

<script>
const QUEUE = /*QUEUE_START*/{queue}/*QUEUE_END*/;
let cursor = 0;
const decisions = JSON.parse(localStorage.getItem("issueRagDecisions") || "{{}}");

function pending() {{ return QUEUE.filter(e => !decisions[e.key]); }}

function render() {{
  const left = pending();
  document.getElementById("done").textContent = Object.keys(decisions).length;
  document.getElementById("progress").style.width =
    (100 * Object.keys(decisions).length / QUEUE.length) + "%";
  const card = document.getElementById("card");
  if (!left.length) {{
    card.innerHTML = '<div class="done">全部完成，' + Object.keys(decisions).length +
      ' 条已记录。<br><span class="meta">决策已实时写入仓库；可以回到对话里继续。</span></div>';
    document.getElementById("idx").textContent = QUEUE.length;
    return;
  }}
  const e = left[0];
  document.getElementById("idx").textContent = QUEUE.length - left.length + 1;
  card.innerHTML =
    '<div class="meta">' + e.item_id + '</div>' +
    '<div class="block q"><div class="label">问题</div>' + esc(e.question) + '</div>' +
    '<div class="block ref"><div class="label">参考答案（维护者的结论）</div>' + esc(e.reference) + '</div>' +
    '<div class="block cand"><div class="label">待评答案</div>' + esc(e.candidate) + '</div>' +
    '<textarea id="note" placeholder="备注（可选，判分歧时有用）"></textarea>' +
    '<div class="row">' +
      '<button class="c" onclick="judge(\\'correct\\')">c · 正确</button>' +
      '<button class="p" onclick="judge(\\'partial\\')">p · 部分正确</button>' +
      '<button class="i" onclick="judge(\\'incorrect\\')">i · 错误</button>' +
      '<button onclick="skip()">s · 跳过</button>' +
    '</div>';
  document.getElementById("note").focus();
}}

function esc(s) {{
  const d = document.createElement("div"); d.textContent = s || ""; return d.innerHTML;
}}

async function judge(verdict) {{
  const e = pending()[0];
  if (!e) return;
  const note = (document.getElementById("note") || {{}}).value || "";
  decisions[e.key] = {{ verdict, note }};
  localStorage.setItem("issueRagDecisions", JSON.stringify(decisions));
  render();
  try {{
    await fetch("/decide", {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify({{ key: e.key, verdict, note }})
    }});
    document.getElementById("warn").textContent = "";
  }} catch (err) {{
    document.getElementById("warn").textContent =
      "保存失败，决策只在浏览器里：" + err;
  }}
}}

function skip() {{
  const e = pending()[0];
  if (!e) return;
  decisions[e.key] = {{ verdict: "skip", note: "" }};
  localStorage.setItem("issueRagDecisions", JSON.stringify(decisions));
  render();
}}

document.addEventListener("keydown", ev => {{
  if (ev.target.tagName === "TEXTAREA") {{
    if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey)) {{ ev.preventDefault(); judge(prompt()); }}
    return;
  }}
  const k = ev.key.toLowerCase();
  if (k === "c") judge("correct");
  else if (k === "p") judge("partial");
  else if (k === "i") judge("incorrect");
  else if (k === "s") skip();
}});

render();
</script>
"""


def _page_for(queue: list[dict[str, Any]]) -> str:
    """Render the review page for a queue.

    Nothing identifying the judge is exposed. Entries are addressed by their index
    rather than by ``review_id``, because that id embeds the system name
    (``closed_book:q12``) and a reviewer reading the page source would learn which
    system produced each answer.
    """
    public = [
        {
            "key": index,
            "item_id": entry["item_id"],
            "question": entry["question"],
            "reference": entry["reference"],
            "candidate": entry["candidate"],
        }
        for index, entry in enumerate(queue)
    ]
    return PAGE.format(total=len(public), queue=json.dumps(public, ensure_ascii=False))


def serve(
    preset: RepoPreset,
    queue_path: Path,
    decisions_path: Path,
    port: int,
    width: int = 96,
) -> int:
    """Serve the blind review page on localhost and write decisions as they happen.

    The browser is the only interface that works here: the agent that set this up
    cannot hand over an interactive stdin, and a file-based workflow would need the
    reviewer to move files around. Decisions POST back, so the JSONL lands in the
    repo while the reviewer clicks.
    """
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    queue = load_jsonl(queue_path)
    if not queue:
        raise SystemExit(f"no queue at {queue_path}; run without --serve first")
    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    page = _page_for(queue).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)

        def do_POST(self) -> None:  # noqa: N802 - stdlib naming
            if self.path != "/decide":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self.send_error(400)
                return
            key = payload.get("key")
            verdict = payload.get("verdict")
            if verdict == "skip":
                self.send_response(204)
                self.end_headers()
                return
            if verdict not in VERDICTS or not isinstance(key, int):
                self.send_error(400)
                return
            if not 0 <= key < len(queue):
                self.send_error(404)
                return
            entry = queue[key]
            with decisions_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "review_id": entry["review_id"],
                            "item_id": entry["item_id"],
                            "system": entry["system"],
                            "human_verdict": verdict,
                            "human_note": payload.get("note") or "",
                            "question": entry["question"],
                            "candidate": entry["candidate"],
                            "judge_verdict": entry["judge_verdict"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            self.send_response(204)
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"queue: {len(queue)} items | decisions -> {decisions_path}")
    print(f"open this in a browser: {url}")
    print("Ctrl+C to stop. Decisions are written as you click.")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        while True:
            threading.Event().wait(1.0)
    except KeyboardInterrupt:
        server.shutdown()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="vite")
    parser.add_argument("--systems", default=DEFAULT_SYSTEMS)
    parser.add_argument("--n", type=int, default=30, help="total items to sample")
    parser.add_argument("--seed", type=int, default=20261029, help="sampling seed")
    parser.add_argument("--review", action="store_true", help="grade the queue in a terminal")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="serve the blind review page on localhost (recommended: works without a TTY)",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--score", action="store_true", help="compare with the judge")
    parser.add_argument("--width", type=int, default=96, help="wrap width")
    parser.add_argument("--force", action="store_true", help="overwrite an existing queue")
    args = parser.parse_args(argv)

    preset = load_preset(args.preset)
    queue_path = preset.processed_dir / QUEUE_FILE
    decisions_path = preset.processed_dir / DECISIONS_FILE

    if args.serve:
        if not queue_path.exists():
            raise SystemExit(f"no queue at {queue_path}; run without --serve first")
        return serve(preset, queue_path, decisions_path, args.port, args.width)
    if args.review:
        if not queue_path.exists():
            raise SystemExit(f"no queue at {queue_path}; run without --review first")
        return review(preset, queue_path, decisions_path, args.width)
    if args.score:
        return score(preset, decisions_path)

    if queue_path.exists() and not args.force:
        existing = load_jsonl(queue_path)
        raise SystemExit(
            f"queue already exists with {len(existing)} items at {queue_path}.\n"
            "Add --force to rebuild it (this does not delete your decisions)."
        )

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    queue = build_queue(preset, systems=systems, total=args.n, seed=args.seed)
    queue_path.write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in queue) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(queue)} items to {queue_path}")
    print(f"  systems: {', '.join(systems)}  |  seed: {args.seed}")
    print(f"  next: python -m issue_rag.human_review --preset {args.preset} --review")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
