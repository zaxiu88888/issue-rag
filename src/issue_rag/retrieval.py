"""四种检索：BM25、dense、RRF 混合、cross-encoder 重排。

Retrieval: lexical (BM25), dense (bge), hybrid fusion, and cross-encoder rerank.

The four configurations exist so the results table can show *where the gain
comes from* rather than just that some pipeline works. Every ablation answer an
interviewer is likely to ask (does BM25 matter? does rerank pay for its
latency?) is a row in that table.

Caching
-------
Dense embeddings dominate runtime, so they are cached under ``data/cache`` keyed
by a hash of the corpus plus the model name. Changing the corpus or the model
invalidates the cache automatically.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .config import CACHE_DIR

CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
TOKEN = re.compile(r"[A-Za-z0-9_.\-/]+")

# Function words add noise to BM25 on short queries without helping ranking.
STOPWORDS = frozenset(
    """a an the is are was were be been being do does did doing have has had
    how what why when where which who whom this that these those it its in on
    at to for of with from by as or and not no if then than there here can
    could should would will i you we they he she my your our their""".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase, split identifiers, and keep technical tokens intact.

    ``streamText`` and ``stream_text`` should both match ``stream text``; version
    strings like ``4.0.12`` must survive as one token because "which version
    fixed it" questions hinge on them.
    """
    tokens: list[str] = []
    for raw in TOKEN.findall(text):
        tok = raw.lower().strip("._-/")
        if not tok:
            continue
        if tok not in STOPWORDS and len(tok) > 1:
            tokens.append(tok)
        # Split camelCase / snake_case so both the compound and parts are indexed.
        for part in CAMEL.sub(" ", raw).replace("_", " ").replace("/", " ").split():
            part = part.lower().strip("._-")
            if part and part not in STOPWORDS and len(part) > 1 and part != tok:
                tokens.append(part)
    return tokens


@dataclass
class Hit:
    doc_id: str
    index: int
    score: float
    issue_number: int
    text: str
    metadata: dict[str, Any]


class Corpus:
    """In-memory view over corpus.jsonl."""

    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self.docs = docs
        self.texts = [d["text"] for d in docs]
        self.issue_numbers = [d["issue_number"] for d in docs]
        self.fingerprint = hashlib.sha1(
            "\n".join(d["doc_id"] for d in docs).encode("utf-8")
        ).hexdigest()[:16]

    def __len__(self) -> int:
        return len(self.docs)

    @classmethod
    def load(cls, path: Path) -> "Corpus":
        docs: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    docs.append(json.loads(line))
        if not docs:
            raise SystemExit(f"empty corpus at {path}; run issue_rag.corpus first")
        return cls(docs)

    def hits(
        self,
        indices: Sequence[int],
        scores: Sequence[float],
        *,
        exclude_issue: int | None,
        top_k: int,
    ) -> list[Hit]:
        """Materialise hits, optionally hiding one issue's own documents."""
        out: list[Hit] = []
        for idx, score in zip(indices, scores):
            if exclude_issue is not None and self.issue_numbers[idx] == exclude_issue:
                continue
            out.append(
                Hit(
                    doc_id=self.docs[idx]["doc_id"],
                    index=int(idx),
                    score=float(score),
                    issue_number=self.issue_numbers[idx],
                    text=self.docs[idx]["text"],
                    metadata=self.docs[idx],
                )
            )
            if len(out) >= top_k:
                break
        return out


class BM25Retriever:
    """Lexical baseline. Cheap, strong on exact identifiers and error strings."""

    def __init__(self, corpus: Corpus) -> None:
        from rank_bm25 import BM25Okapi

        self.corpus = corpus
        tokenised = [tokenize(text) for text in corpus.texts]
        self.bm25 = BM25Okapi(tokenised)

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        exclude_issue: int | None = None,
        pool: int = 200,
    ) -> list[Hit]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        order = np.argsort(scores)[::-1][:pool]
        return self.corpus.hits(
            order, scores[order], exclude_issue=exclude_issue, top_k=top_k
        )


class DenseRetriever:
    """Sentence-transformers bi-encoder over the same corpus."""

    def __init__(
        self,
        corpus: Corpus,
        model_path: str,
        *,
        device: str | None = None,
        batch_size: int = 64,
        cache_dir: Path | None = None,
        query_prefix: str = "",
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.corpus = corpus
        self.query_prefix = query_prefix
        self.model = SentenceTransformer(model_path, device=device)
        self.batch_size = batch_size
        self.cache_dir = cache_dir or CACHE_DIR
        self.model_slug = re.sub(r"[^\w.-]+", "_", model_path)[-60:]
        self.embeddings = self._load_or_build()

    def _cache_paths(self) -> tuple[Path, Path]:
        base = self.cache_dir / "embeddings"
        base.mkdir(parents=True, exist_ok=True)
        stem = f"{self.model_slug}-{self.corpus.fingerprint}"
        return base / f"{stem}.npy", base / f"{stem}.json"

    def _load_or_build(self) -> np.ndarray:
        npy_path, meta_path = self._cache_paths()
        if npy_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            if meta.get("docs") == len(self.corpus):
                print(f"  embedding cache hit: {npy_path.name}")
                return np.load(npy_path)

        started = time.time()
        print(f"  embedding {len(self.corpus)} docs with {self.model_slug} ...")
        vectors = self.model.encode(
            self.corpus.texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        np.save(npy_path, vectors)
        meta_path.write_text(
            json.dumps({"docs": len(self.corpus), "model": self.model_slug}),
            encoding="utf-8",
        )
        print(f"  embedded in {time.time() - started:.1f}s -> {npy_path.name}")
        return vectors

    def encode_query(self, query: str) -> np.ndarray:
        text = f"{self.query_prefix}{query}" if self.query_prefix else query
        vec = self.model.encode(
            [text], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return vec.astype(np.float32)[0]

    def scores(self, query: str) -> np.ndarray:
        return self.embeddings @ self.encode_query(query)

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        exclude_issue: int | None = None,
        pool: int = 200,
    ) -> list[Hit]:
        scores = self.scores(query)
        order = np.argsort(scores)[::-1][:pool]
        return self.corpus.hits(
            order, scores[order], exclude_issue=exclude_issue, top_k=top_k
        )


def reciprocal_rank_fusion(
    rankings: Iterable[Sequence[int]], k: int = 60, pool: int = 200
) -> list[tuple[int, float]]:
    """倒数排名融合：只看名次、不比较分数，让 BM25 与余弦分可以混合。

    Fuse multiple ranked index lists without needing comparable scores."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking[:pool]):
            fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


class HybridRetriever:
    """BM25 + dense fused with RRF, optionally reordered by a cross-encoder."""

    def __init__(
        self,
        corpus: Corpus,
        *,
        dense: DenseRetriever,
        bm25: BM25Retriever | None = None,
        reranker_path: str | None = None,
        device: str | None = None,
        rrf_k: int = 60,
        candidate_pool: int = 40,
    ) -> None:
        self.corpus = corpus
        self.dense = dense
        self.bm25 = bm25 or BM25Retriever(corpus)
        self.rrf_k = rrf_k
        self.candidate_pool = candidate_pool
        self.reranker = None
        if reranker_path:
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(reranker_path, device=device, max_length=512)

    def _fuse(
        self, query: str, *, exclude_issue: int | None, pool: int
    ) -> list[tuple[int, float]]:
        bm25_hits = self.bm25.search(query, top_k=pool, exclude_issue=exclude_issue, pool=pool)
        dense_scores = self.dense.scores(query)
        dense_order = np.argsort(dense_scores)[::-1][: pool * 2]

        rankings: list[list[int]] = [[h.index for h in bm25_hits]]
        dense_list: list[int] = []
        for idx in dense_order:
            if exclude_issue is not None and self.corpus.issue_numbers[int(idx)] == exclude_issue:
                continue
            dense_list.append(int(idx))
            if len(dense_list) >= pool:
                break
        rankings.append(dense_list)
        return reciprocal_rank_fusion(rankings, k=self.rrf_k, pool=pool)

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        exclude_issue: int | None = None,
        pool: int = 200,
    ) -> list[Hit]:
        fused = self._fuse(query, exclude_issue=exclude_issue, pool=pool)

        if self.reranker is not None:
            candidates = fused[: self.candidate_pool]
            pairs = [(query, self.corpus.texts[idx]) for idx, _ in candidates]
            scores = self.reranker.predict(pairs)
            reranked = sorted(
                ((idx, float(score)) for (idx, _), score in zip(candidates, scores)),
                key=lambda kv: kv[1],
                reverse=True,
            )
            indices = [idx for idx, _ in reranked]
            values = [score for _, score in reranked]
        else:
            indices = [idx for idx, _ in fused]
            values = [score for _, score in fused]

        return self.corpus.hits(indices, values, exclude_issue=exclude_issue, top_k=top_k)


def build_retriever(
    mode: str,
    corpus: Corpus,
    *,
    dense_model: str,
    reranker_model: str | None,
    device: str | None,
) -> Any:
    """Factory shared by the evaluation runner and the demo app."""
    if mode == "bm25":
        return BM25Retriever(corpus)
    if mode == "dense":
        return DenseRetriever(corpus, dense_model, device=device)
    if mode in {"hybrid", "hybrid_rerank"}:
        dense = DenseRetriever(corpus, dense_model, device=device)
        return HybridRetriever(
            corpus,
            dense=dense,
            reranker_path=reranker_model if mode == "hybrid_rerank" else None,
            device=device,
        )
    raise SystemExit(f"unknown retrieval mode {mode!r}")
