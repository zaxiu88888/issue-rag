"""把模型权重下载到 data/models（默认走 ModelScope）。

Download model weights into ``data/models``.

Why ModelScope by default: ``huggingface.co`` is unreachable from some
networks (including the machine this was built on) while
``www.modelscope.cn`` is. ModelScope hosts mirrors of the same weights under the
same architecture, so the rest of the pipeline is unaffected.

    python -m issue_rag.models                 # default trio
    python -m issue_rag.models --only generator
    python -m issue_rag.models --backend hf    # if you do have HF access
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import PROJECT_ROOT

MODELS_DIR = PROJECT_ROOT / "data" / "models"

DEFAULT_MODELS = {
    # Local id (same string used as --model) : (modelscope id, huggingface id)
    "generator": (
        "data/models/Qwen2.5-3B-Instruct",
        "qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
    ),
    "dense": (
        "data/models/bge-base-en-v1.5",
        "BAAI/bge-base-en-v1.5",
        "BAAI/bge-base-en-v1.5",
    ),
    "reranker": (
        "data/models/bge-reranker-base",
        "BAAI/bge-reranker-base",
        "BAAI/bge-reranker-base",
    ),
}


def download(local_id: str, ms_id: str, hf_id: str, backend: str) -> Path:
    target = PROJECT_ROOT / local_id
    if target.exists() and any(target.iterdir()):
        print(f"  {local_id}: already present, skipping")
        return target
    target.mkdir(parents=True, exist_ok=True)

    if backend == "hf":
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=hf_id, local_dir=str(target))
    else:
        from modelscope import snapshot_download

        snapshot_download(ms_id, local_dir=str(target))
    print(f"  {local_id}: done")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download model weights")
    parser.add_argument("--backend", choices=("modelscope", "hf"), default="modelscope")
    parser.add_argument(
        "--only",
        default=",".join(DEFAULT_MODELS),
        help="comma separated subset of: " + ", ".join(DEFAULT_MODELS),
    )
    args = parser.parse_args(argv)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    wanted = [name.strip() for name in args.only.split(",") if name.strip()]
    for name in wanted:
        if name not in DEFAULT_MODELS:
            raise SystemExit(f"unknown model {name!r}; choose from {list(DEFAULT_MODELS)}")
        local_id, ms_id, hf_id = DEFAULT_MODELS[name]
        print(f"[{name}] {ms_id if args.backend == 'modelscope' else hf_id}")
        download(local_id, ms_id, hf_id, args.backend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
