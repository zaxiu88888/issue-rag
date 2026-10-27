"""Project paths and repository presets.

The pipeline is repo-agnostic on purpose: every stage reads its target from a
preset, so swapping the domain under study is a one-line config edit rather
than a code change.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

import yaml

# src/issue_rag/config.py -> src/issue_rag -> src -> <project root>
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]

CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = DATA_DIR / "results"

PRESETS_PATH = CONFIG_DIR / "presets.yaml"


@dataclasses.dataclass(frozen=True)
class RepoPreset:
    """A target repository for the support-assistant corpus."""

    key: str
    repo: str
    display_name: str
    language: str = ""
    notes: str = ""

    @property
    def slug(self) -> str:
        return self.repo.replace("/", "__")

    @property
    def raw_dir(self) -> pathlib.Path:
        return RAW_DIR / self.slug

    @property
    def processed_dir(self) -> pathlib.Path:
        return PROCESSED_DIR / self.slug

    def ensure_dirs(self) -> None:
        for path in (self.raw_dir, self.processed_dir, CACHE_DIR, RESULTS_DIR):
            path.mkdir(parents=True, exist_ok=True)


def load_presets(path: pathlib.Path | None = None) -> dict[str, RepoPreset]:
    """Read every preset from presets.yaml, keyed by preset name."""
    path = path or PRESETS_PATH
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: dict[str, RepoPreset] = {}
    for key, value in (raw.get("presets") or {}).items():
        out[key] = RepoPreset(
            key=key,
            repo=value["repo"],
            display_name=value.get("display_name", value["repo"]),
            language=value.get("language", ""),
            notes=value.get("notes", ""),
        )
    return out


def load_preset(key: str) -> RepoPreset:
    presets = load_presets()
    if key not in presets:
        available = ", ".join(sorted(presets)) or "<none>"
        raise SystemExit(f"unknown preset {key!r}; available: {available}")
    preset = presets[key]
    preset.ensure_dirs()
    return preset
