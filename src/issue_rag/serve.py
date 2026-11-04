"""极简浏览器演示：提问 → 答案 + 引用的 issue 线程。

Minimal browser demo: ask a question, get an answer plus the issue threads it used.

Deliberately dependency-free (``http.server``) so the demo cannot fail to start
and adds nothing to the install. A richer UI belongs in a separate front-end
project; this one exists to make the system inspectable in ninety seconds.

    python -m issue_rag.serve --preset vite

Then open http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import html
import re
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .config import RepoPreset, load_env_file, load_preset
from .corpus import load_jsonl
from .evaluate import RAG_SYSTEM_PROMPT, build_context
from .generate import resolve_model
from .retrieval import Corpus, build_retriever

STATE: dict[str, Any] = {}

PAGE = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>issue-rag</title>
<style>
  :root {{ --ink:#1b1f24; --muted:#6b7280; --line:#e5e7eb; --bg:#fbfbfc; }}
  * {{ box-sizing: border-box; }}
  body {{ font: 16px/1.65 -apple-system, "Segoe UI", Roboto, sans-serif;
         max-width: 780px; margin: 0 auto; padding: 3rem 1.25rem 5rem;
         color: var(--ink); background:#fff; }}
  h1 {{ font-size: 1.15rem; font-weight: 600; margin: 0 0 .35rem; }}
  h2 {{ font-size: .8rem; font-weight: 600; letter-spacing: .08em;
        text-transform: uppercase; color: var(--muted); margin: 2rem 0 .8rem; }}
  p.lede {{ color: var(--muted); font-size: .9rem; margin: 0 0 1.6rem; }}
  form {{ display: flex; gap: .5rem; margin: 0 0 .9rem; }}
  input[type=text] {{ flex: 1; padding: .7rem .9rem; font-size: 1rem;
                      border: 1px solid var(--line); border-radius: 8px; }}
  input[type=text]:focus {{ outline: 2px solid #111; outline-offset: 1px; }}
  button {{ padding: .7rem 1.2rem; font-size: 1rem; border: 0; border-radius: 8px;
            background: #111; color: #fff; cursor: pointer; }}
  .note {{ color: var(--muted); font-size: .82rem; margin: 0 0 2rem;
           border-left: 2px solid var(--line); padding-left: .8rem; }}
  .q {{ font-weight: 600; margin: 0 0 .8rem; }}
  .answer {{ background: var(--bg); border: 1px solid var(--line); border-radius: 10px;
             padding: 1.1rem 1.3rem; font-size: .97rem; }}
  .answer p {{ margin: 0 0 .8rem; }}
  .answer p:last-child {{ margin-bottom: 0; }}
  .answer code {{ background: #eef0f3; padding: .1rem .35rem; border-radius: 4px;
                  font-size: .88em; font-family: ui-monospace, Consolas, monospace; }}
  .answer a.cite {{ color: #0b5cad; text-decoration: none; font-size: .78em;
                    vertical-align: super; padding: 0 .1em; font-weight: 600; }}
  .answer a.cite:hover {{ text-decoration: underline; }}
  ol.src {{ list-style: none; margin: 0; padding: 0;
            counters: none; }}
  li.src {{ border: 1px solid var(--line); border-radius: 10px; padding: .85rem 1rem;
            margin: 0 0 .6rem; font-size: .9rem; }}
  li.src:target {{ border-color: #111; box-shadow: 0 0 0 2px #1111; }}
  .src-head {{ display: flex; gap: .5rem; align-items: baseline; flex-wrap: wrap;
               margin-bottom: .35rem; }}
  .badge {{ font-size: .72rem; font-weight: 700; color: #0b5cad; background: #e8f0fb;
            border-radius: 4px; padding: .05rem .35rem; }}
  .src-title {{ font-weight: 600; }}
  .src-title a {{ color: inherit; text-decoration: none; }}
  .src-title a:hover {{ text-decoration: underline; }}
  .score {{ color: var(--muted); font-size: .76rem; margin-left: auto; }}
  .snippet {{ color: #374151; margin: 0; }}
  .snippet code {{ background: #eef0f3; padding: .05rem .3rem; border-radius: 4px;
                   font-family: ui-monospace, Consolas, monospace; font-size: .88em; }}
  details {{ margin-top: .45rem; }}
  details summary {{ cursor: pointer; color: var(--muted); font-size: .78rem; }}
  .err {{ border-color: #f0c0c0; background: #fff5f5; }}
</style>
<h1>issue-rag &mdash; {repo}</h1>
<p class="lede">Answers are grounded in retrieved issue threads and cite them by number.
Nothing here is generated without a source.</p>
<form method="get">
  <input type="text" name="q" value="{query}" placeholder="e.g. why does my css update twice" autofocus>
  <button type="submit">Ask</button>
</form>
<p class="note"><strong>Ask in English.</strong> The corpus, the BM25 tokenizer and the
embedding model (bge-base-en-v1.5) are English-only, so a question in another
language retrieves noise and the answer will be a refusal rather than a wrong fact.</p>
{body}
"""

HIT = """<li class="src" id="src-{rank}">
  <div class="src-head">
    <span class="badge">{rank}</span>
    <span class="src-title"><a href="{url}" target="_blank" rel="noreferrer">{title}</a></span>
    <span class="score">{score:.3f}</span>
  </div>
  <p class="snippet">{short}{ellipsis}</p>
  <details><summary>full comment</summary><p class="snippet">{full}</p></details>
</li>"""


def _inline_md(text: str) -> str:
    """Render the little markdown the model actually emits, and nothing more.

    The generator answers in markdown (`code`, **bold**, blank-line paragraphs).
    The page used to escape it all and rely on ``white-space: pre-wrap``, which is
    why every answer looked like a wall of asterisks and backticks. Escaping still
    happens first -- the model output is untrusted, and this is not a markdown
    parser, just enough syntax to make prose read as prose.
    """
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*\n]+)\*\*", r"<strong>\1</strong>", escaped)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", escaped) if p.strip()]
    return "".join(f"<p>{p.replace(chr(10), '<br>')}</p>" for p in paragraphs)


def _cite_links(body: str, count: int) -> str:
    """Turn ``[3]`` into a jump link to source card 3.

    The citations were plain text, so a reader had to scroll and count cards to
    find out what ``[3]`` referred to. They are anchors now.
    """

    def replace(match: re.Match[str]) -> str:
        index = int(match.group(1))
        if 1 <= index <= count:
            return f'<a class="cite" href="#src-{index}">[{index}]</a>'
        return match.group(0)

    return re.sub(r"\[(\d{1,2})\]", replace, body)


def answer_question(question: str) -> tuple[str, list[dict[str, Any]]]:
    preset: RepoPreset = STATE["preset"]
    retriever = STATE["retriever"]
    model = STATE["model"]

    hits = retriever.search(question, top_k=STATE["top_k"])
    context = build_context(hits)
    messages = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT.format(repo=preset.repo)},
        {
            "role": "user",
            "content": f"EXCERPTS:\n{context}\n\nQUESTION:\n{question}",
        },
    ]
    result = model.chat(messages, max_new_tokens=STATE["max_new_tokens"])
    payload = [
        {
            "number": hit.issue_number,
            "score": hit.score,
            "url": hit.metadata.get("url") or "",
            # The chunk text starts with "Issue #N: <title>", which the card now
            # shows as its own title, so strip the prefix instead of repeating it.
            "snippet": _strip_title_prefix(hit.text),
            "title": hit.metadata.get("title") or f"issue #{hit.issue_number}",
        }
        for hit in hits
    ]
    return result.text, payload


def _strip_title_prefix(text: str) -> str:
    return re.sub(r"^Issue #\d+:\s*[^\n]*\n+", "", text).strip() or text.strip()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:  # quieter console
        pass

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        query = (params.get("q") or [""])[0].strip()

        body = ""
        if query:
            try:
                answer, hits = answer_question(query)
                cards = []
                for i, h in enumerate(hits):
                    full = _inline_md(h["snippet"])
                    plain = h["snippet"]
                    short = _inline_md(plain[:180].rsplit(" ", 1)[0]) if len(plain) > 180 else full
                    cards.append(
                        HIT.format(
                            rank=i + 1,
                            number=h["number"],
                            score=h["score"],
                            url=html.escape(h["url"]),
                            title=html.escape(h["title"]),
                            short=short,
                            ellipsis="&hellip;" if len(plain) > 180 else "",
                            full=full,
                        )
                    )
                body = (
                    f'<p class="q">{html.escape(query)}</p>'
                    f'<div class="answer">{_cite_links(_inline_md(answer), len(hits))}</div>'
                    f'<h2>Sources &mdash; {len(hits)} issue threads</h2>'
                    f'<ol class="src">{"".join(cards)}</ol>'
                )
            except Exception as exc:  # noqa: BLE001 - surface errors in the page
                body = f'<div class="answer err">error: {html.escape(str(exc))}</div>'

        page = PAGE.format(
            repo=html.escape(STATE["preset"].display_name),
            query=html.escape(query),
            body=body,
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(page.encode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the demo UI")
    parser.add_argument("--preset", default="vite")
    parser.add_argument(
        "--provider",
        default="local",
        choices=("local", "deepseek", "openai", "siliconflow", "ollama"),
    )
    parser.add_argument("--endpoint", default="", help="explicit OpenAI-compatible base URL")
    parser.add_argument("--model", default="", help="override the provider's default model")
    parser.add_argument("--local-model", default="data/models/Qwen2.5-3B-Instruct")
    parser.add_argument("--dense-model", default="data/models/bge-base-en-v1.5")
    parser.add_argument("--reranker-model", default="data/models/bge-reranker-base")
    parser.add_argument("--retrieval", default="hybrid_rerank")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    load_env_file()
    preset = load_preset(args.preset)
    corpus = Corpus.load(preset.processed_dir / "corpus.jsonl")
    print(f"corpus: {len(corpus)} documents")

    STATE.update(
        {
            "preset": preset,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "retriever": build_retriever(
                args.retrieval,
                corpus,
                dense_model=args.dense_model,
                reranker_model=args.reranker_model,
                device=None,
            ),
            "model": resolve_model(
                provider=args.provider,
                endpoint=args.endpoint,
                model=args.model,
                local_default=args.local_model,
            ),
        }
    )

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"serving on http://127.0.0.1:{args.port}")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        while True:
            threading.Event().wait(1.0)
    except KeyboardInterrupt:
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
