"""Minimal browser demo: ask a question, get an answer plus the issue threads it used.

Deliberately dependency-free (``http.server``) so the demo cannot fail to start
and adds nothing to the install. A richer UI belongs in a separate front-end
project; this one exists to make the system inspectable in ninety seconds.

    python -m issue_rag.serve --preset vite

Then open http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import html
import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .config import RepoPreset, load_preset
from .corpus import load_jsonl
from .evaluate import RAG_SYSTEM_PROMPT, build_context
from .generate import LocalChatModel
from .retrieval import Corpus, build_retriever

STATE: dict[str, Any] = {}

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>issue-rag</title>
<style>
  body {{ font: 15px/1.6 -apple-system, "Segoe UI", sans-serif; max-width: 860px;
         margin: 3rem auto; padding: 0 1rem; color: #222; }}
  h1 {{ font-size: 1.3rem; }}
  form {{ display: flex; gap: .5rem; margin: 1.5rem 0; }}
  input[type=text] {{ flex: 1; padding: .6rem .8rem; font-size: 1rem;
                      border: 1px solid #ccc; border-radius: 6px; }}
  button {{ padding: .6rem 1.1rem; font-size: 1rem; border: 0; border-radius: 6px;
            background: #222; color: #fff; cursor: pointer; }}
  .answer {{ background: #f6f7f9; border-left: 3px solid #222; padding: 1rem;
             white-space: pre-wrap; border-radius: 0 6px 6px 0; }}
  .hit {{ border: 1px solid #e3e3e3; border-radius: 6px; padding: .7rem;
          margin: .6rem 0; font-size: .88rem; }}
  .hit a {{ color: #0a58ca; text-decoration: none; }}
  .muted {{ color: #777; font-size: .82rem; }}
  details summary {{ cursor: pointer; }}
</style>
<h1>issue-rag &mdash; {repo}</h1>
<p class="muted">Answers are grounded in retrieved issue threads and cite them by number.
Nothing here is generated without a source.</p>
<form method="get">
  <input type="text" name="q" value="{query}" placeholder="e.g. why does my css update twice" autofocus>
  <button type="submit">Ask</button>
</form>
{body}
"""

HIT = """<div class="hit">
  <div class="muted">[{rank}] issue #{number} &middot; score {score:.3f} &middot;
    <a href="{url}" target="_blank" rel="noreferrer">open thread</a></div>
  <div>{snippet}</div>
</div>"""


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
            "snippet": hit.text[:400],
        }
        for hit in hits
    ]
    return result.text, payload


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
                rendered = "".join(
                    HIT.format(
                        rank=i + 1,
                        number=h["number"],
                        score=h["score"],
                        url=html.escape(h["url"]),
                        snippet=html.escape(h["snippet"]),
                    )
                    for i, h in enumerate(hits)
                )
                body = (
                    f'<div class="answer">{html.escape(answer)}</div>'
                    f"<h2 style='font-size:1rem;margin-top:1.6rem'>Retrieved threads</h2>"
                    f"{rendered}"
                )
            except Exception as exc:  # noqa: BLE001 - surface errors in the page
                body = f'<div class="answer">error: {html.escape(str(exc))}</div>'

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
    parser.add_argument("--model", default="data/models/Qwen2.5-3B-Instruct")
    parser.add_argument("--dense-model", default="data/models/bge-base-en-v1.5")
    parser.add_argument("--reranker-model", default="data/models/bge-reranker-base")
    parser.add_argument("--retrieval", default="hybrid_rerank")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

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
            "model": LocalChatModel(args.model),
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
