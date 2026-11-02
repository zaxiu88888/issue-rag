"""Unit tests for the parts that can run with no network, GPU, or model weights.

The heavy stages (harvest, embed, generate) are not covered here on purpose:
they are expensive and flaky to test hermetically. What *is* covered is every
piece of logic that could silently corrupt the experiment -- chunking, the BM25
tokenizer, rank fusion, eval-set filtering, and judge output parsing.

CI runs exactly this file, which is why it must stay dependency-light.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from issue_rag.corpus import chunk_text, clean_text, is_bot  # noqa: E402
from issue_rag.evalset import LOW_VALUE_ANSWER, answer_features, collect_candidates  # noqa: E402
from issue_rag.judge import parse_json_object  # noqa: E402
from issue_rag.retrieval import reciprocal_rank_fusion, tokenize  # noqa: E402


# ----------------------------------------------------------------- chunking


def test_chunk_short_text_is_single_chunk():
    assert chunk_text("hello world") == ["hello world"]


def test_chunk_respects_budget_and_keeps_all_paragraphs():
    paragraphs = [f"paragraph {i} " + "x" * 400 for i in range(12)]
    text = "\n\n".join(paragraphs)
    chunks = chunk_text(text, max_chars=1200, overlap=0)
    assert len(chunks) > 1
    for chunk in chunks:
        # The packing step may append one paragraph that pushes slightly over
        # budget; it must never be wildly over.
        assert len(chunk) <= 1800
    joined = " ".join(chunks)
    for i in range(12):
        assert f"paragraph {i} " in joined


def test_chunk_splits_single_oversized_paragraph():
    text = "y" * 5000
    chunks = chunk_text(text, max_chars=1000, overlap=100)
    assert len(chunks) >= 5
    assert all(len(c) <= 1000 for c in chunks)


def test_chunk_empty_input():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_clean_text_strips_html_comments_and_collapses_blank_lines():
    raw = "a\r\n\r\n\r\n\r\nb<!-- hidden -->\n"
    cleaned = clean_text(raw)
    assert "hidden" not in cleaned
    assert "\r" not in cleaned
    assert "\n\n\n" not in cleaned


@pytest.mark.parametrize(
    "login,expected",
    [
        ("github-actions[bot]", True),
        ("dependabot", True),
        ("vercel", True),
        ("sarah", False),
        (None, False),
    ],
)
def test_is_bot(login, expected):
    assert is_bot(login) is expected


# ----------------------------------------------------------------- tokenizer


def test_tokenize_splits_camel_case_and_snake_case():
    tokens = tokenize("streamText convert_to_model_messages")
    assert "streamtext" in tokens
    assert "stream" in tokens
    assert "text" in tokens
    assert "convert_to_model_messages" in tokens or "convert" in tokens


def test_tokenize_keeps_version_strings_whole():
    # "which version fixed this" questions hinge on the version surviving as a
    # single token; splitting it would destroy the strongest BM25 signal.
    tokens = tokenize("fixed in 4.0.12")
    assert "4.0.12" in tokens


def test_tokenize_drops_stopwords_and_empty():
    tokens = tokenize("how do I use the tool")
    assert "the" not in tokens
    assert "how" not in tokens
    assert "tool" in tokens
    assert tokenize("") == []


# ---------------------------------------------------------------- rank fusion


def test_rrf_rewards_agreement_between_lists():
    # Document 7 appears high in both lists, so it must win.
    fused = reciprocal_rank_fusion([[1, 7, 3], [7, 2, 1]], k=60)
    assert fused[0][0] == 7


def test_rrf_handles_single_list():
    fused = reciprocal_rank_fusion([[5, 6, 7]], k=60)
    assert [idx for idx, _ in fused] == [5, 6, 7]


def test_rrf_scores_descend():
    fused = reciprocal_rank_fusion([[1, 2], [2, 1]], k=60)
    scores = [score for _, score in fused]
    assert scores == sorted(scores, reverse=True)


# ------------------------------------------------------------------ evalset


def _issue(number=1, title="streamText re-validates history on every step", **kw):
    return {
        "number": number,
        "title": title,
        "body": "body",
        "state": "closed",
        "created_at": "2026-10-01T00:00:00Z",
        "html_url": f"https://github.com/o/r/issues/{number}",
        "labels": kw.get("labels", []),
        "is_pull_request": kw.get("is_pull_request", False),
    }


def _comment(cid, number=1, association="MEMBER", author="maintainer", body=None):
    return {
        "id": cid,
        "issue_number": number,
        "author": author,
        "author_association": association,
        "body": body or ("Use convertToModelMessages before streaming. " * 6),
        "created_at": "2026-10-02T00:00:00Z",
        "reactions": 0,
        "html_url": "u",
    }


def test_collect_candidates_requires_a_maintainer_answer():
    issues = [_issue(1)]
    comments = [_comment(10, association="NONE", author="random_user")]
    assert collect_candidates(issues, comments) == []


def test_collect_candidates_drops_short_answers():
    issues = [_issue(1)]
    comments = [_comment(10, body="fixed in 5.0.1")]
    assert collect_candidates(issues, comments) == []


def test_collect_candidates_drops_prs_and_duplicates_label():
    assert collect_candidates([_issue(1, is_pull_request=True)], [_comment(10)]) == []
    assert collect_candidates([_issue(1, labels=["duplicate"])], [_comment(10)]) == []


def test_collect_candidates_drops_duplicate_titles():
    issues = [_issue(1), _issue(2, title="streamText re-validates history on every step ")]
    comments = [_comment(10, 1), _comment(11, 2)]
    assert len(collect_candidates(issues, comments)) == 1


def test_collect_candidates_accepts_a_good_pair():
    issues = [_issue(1)]
    comments = [_comment(10)]
    cands = collect_candidates(issues, comments)
    assert len(cands) == 1
    assert cands[0]["query"] == issues[0]["title"]
    assert cands[0]["query_mode"] == "title"
    assert cands[0]["issue_number"] == 1
    assert cands[0]["reference_comment_ids"] == [10]


def test_low_value_answer_matches_triage_boilerplate():
    assert LOW_VALUE_ANSWER.search("Duplicate of #123")
    assert LOW_VALUE_ANSWER.search("closing as stale")
    assert not LOW_VALUE_ANSWER.search("You need to call convertToModelMessages first")


def test_answer_features_flags_version_and_code():
    feats = answer_features("Fixed in 4.0.12:\n```ts\nfoo()\n```\nsee `a/b.ts`")
    assert feats["has_version"]
    assert feats["has_code_block"]
    assert feats["has_file_path"]


# --------------------------------------------------------------- judge parse


def test_parse_json_object_plain():
    assert parse_json_object('{"verdict": "correct"}') == {"verdict": "correct"}


def test_parse_json_object_wrapped_in_prose():
    raw = 'Sure, here is my grade:\n{"verdict": "partial", "fabricated": false}\nHope it helps.'
    parsed = parse_json_object(raw)
    assert parsed["verdict"] == "partial"


def test_parse_json_object_single_quotes():
    assert parse_json_object("{'verdict': 'incorrect'}")["verdict"] == "incorrect"


def test_parse_json_object_returns_none_on_garbage():
    assert parse_json_object("no json at all") is None
    assert parse_json_object("") is None


# ------------------------------------------------------------ harvest helpers


def test_issue_number_from_comment_url():
    from issue_rag.fetch import _issue_number_from_url

    assert _issue_number_from_url("https://api.github.com/repos/vitejs/vite/issues/23889") == 23889
    assert _issue_number_from_url("https://api.github.com/repos/vitejs/vite/issues/23889/") == 23889
    assert _issue_number_from_url("https://api.github.com/repos/o/r/issues/notanumber") is None
    assert _issue_number_from_url(None) is None


def test_since_date_is_in_the_past():
    from issue_rag.fetch import _since_date

    import datetime as dt

    months = 12
    value = dt.date.fromisoformat(_since_date(months))
    age_days = (dt.date.today() - value).days
    assert 350 <= age_days <= 380


def test_build_documents_drops_bot_comments_but_keeps_humans():
    """Bot floods made 84% of one repository's comment sample useless."""
    from issue_rag.corpus import build_documents

    issues = [_issue(1)]
    comments = [
        _comment(10, author="ai-sdk-factory[bot]"),
        _comment(11, author="sapphi-red", association="MEMBER"),
    ]
    kept = list(build_documents(issues, comments, "o/r", drop_bots=True))
    comment_docs = [d for d in kept if d["kind"] == "issue_comment"]
    assert len(comment_docs) == 1
    assert comment_docs[0]["author"] == "sapphi-red"

    kept_all = list(build_documents(issues, comments, "o/r", drop_bots=False))
    assert len([d for d in kept_all if d["kind"] == "issue_comment"]) == 2


def test_build_documents_prefixes_issue_title():
    """Chunk-level text often loses its subject; the prefix restores it."""
    from issue_rag.corpus import build_documents

    issues = [_issue(7, title="build fails on Node 18")]
    comments = [_comment(70, number=7, association="MEMBER")]
    for doc in build_documents(issues, comments, "o/r"):
        assert "build fails on Node 18" in doc["text"]


def test_build_documents_ignores_comments_for_unknown_issues():
    """Comments from the repo-wide feed often belong to issues not harvested."""
    from issue_rag.corpus import build_documents

    issues = [_issue(1)]
    comments = [_comment(99, number=424242)]
    docs = [d for d in build_documents(issues, comments, "o/r") if d["kind"] == "issue_comment"]
    assert docs == []


# --------------------------------------------------------------- provider config


def test_resolve_model_requires_a_key(monkeypatch):
    from issue_rag.generate import MissingCredential, resolve_model

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(MissingCredential):
        resolve_model(provider="deepseek")


def test_resolve_model_uses_provider_defaults(monkeypatch):
    from issue_rag.generate import OpenAICompatChatModel, resolve_model

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    model = resolve_model(provider="deepseek")
    assert isinstance(model, OpenAICompatChatModel)
    assert model.base_url == "https://api.deepseek.com/v1"
    assert model.name == "deepseek-chat"
    assert model.api_key == "sk-test"


def test_resolve_model_explicit_endpoint_wins(monkeypatch):
    from issue_rag.generate import resolve_model

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    model = resolve_model(
        provider="deepseek", endpoint="http://127.0.0.1:8000/v1", model="local-served"
    )
    assert model.base_url == "http://127.0.0.1:8000/v1"
    assert model.name == "local-served"


def test_resolve_model_rejects_unknown_provider():
    from issue_rag.generate import resolve_model

    with pytest.raises(SystemExit):
        resolve_model(provider="not-a-provider")


def test_load_env_file_does_not_override_existing(monkeypatch):
    """Deliberately avoids pytest's tmp_path.

    Some sandboxed Windows environments refuse to create files under the pytest
    temp root (a ``mkdtemp`` directory), which would fail the suite for
    environmental reasons rather than code reasons. A directory beside the test
    file is always writable and behaves the same everywhere.
    """
    import os
    import shutil

    from issue_rag.config import load_env_file

    work = Path(__file__).resolve().parent / "_envfile_tmp"
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    env_file = work / ".env"
    env_file.write_text(
        "A_NEW_KEY=from-file\nA_SET_KEY=from-file\n# comment\n\nB=2\n", encoding="utf-8"
    )
    monkeypatch.delenv("A_NEW_KEY", raising=False)
    monkeypatch.setenv("A_SET_KEY", "from-env")
    monkeypatch.delenv("B", raising=False)

    try:
        loaded = load_env_file(env_file)

        assert os.environ["A_NEW_KEY"] == "from-file"
        assert os.environ["A_SET_KEY"] == "from-env", "existing environment must win"
        assert os.environ["B"] == "2"
        assert "A_SET_KEY" not in loaded
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_fetch_reads_the_token_from_env_file_before_building_the_client(monkeypatch):
    """Regression: a docs-visible credential has to actually be read.

    ``GITHUB_TOKEN`` lives in ``.env`` exactly like ``DEEPSEEK_API_KEY``, but the
    fetch stage was the one entry point that never called ``load_env_file``. A
    token sitting in ``.env`` was therefore ignored and every harvest silently
    ran on the 60/hour anonymous budget -- which is what made widening the
    corpus look impossible. Pin the ordering, not just the call.
    """
    from issue_rag import fetch as fetch_mod

    calls: list[str] = []

    def fake_load_env_file(*args, **kwargs):
        calls.append("env")
        return ["GITHUB_TOKEN"]

    class FakeClient:
        def __init__(self, *args, **kwargs):
            calls.append("client")
            self.stats = type("S", (), {"requests": 0, "cache_hits": 0})()

        def describe_quota(self):
            return "authenticated: core 5000/5000"

    monkeypatch.setattr(fetch_mod, "load_env_file", fake_load_env_file)
    monkeypatch.setattr(fetch_mod, "GitHubClient", FakeClient)
    monkeypatch.setattr(fetch_mod, "harvest_issues", lambda *a, **k: None)

    rc = fetch_mod.main(["issues", "--preset", "vite"])

    assert rc == 0
    assert calls == ["env", "client"], (
        "the .env token must be loaded strictly before the client reads the "
        f"environment, got {calls}"
    )


# ------------------------------------------------------- generator resilience


def _fake_completion() -> bytes:
    import json

    return json.dumps(
        {
            "choices": [{"message": {"content": "the answer"}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 4},
        }
    ).encode("utf-8")


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return _fake_completion()


def test_openai_compat_retries_transient_failures(monkeypatch):
    """A single hung socket once cost 1,031 seconds; retries must absorb blips."""
    import urllib.error
    import urllib.request

    from issue_rag.generate import OpenAICompatChatModel

    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.URLError("transient blip")
        return _FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    model = OpenAICompatChatModel(
        "http://example.invalid/v1",
        "m",
        api_key="k",
        max_attempts=3,
        retry_backoff_s=0.0,
        timeout_s=1,
    )
    result = model.chat([{"role": "user", "content": "hi"}])
    assert result.text == "the answer"
    assert calls["n"] == 3
    assert result.prompt_tokens == 11


def test_openai_compat_raises_generation_error_after_exhausting_attempts(monkeypatch):
    import urllib.error
    import urllib.request

    from issue_rag.generate import GenerationError, OpenAICompatChatModel

    calls = {"n": 0}

    def always_fail(req, timeout=None):
        calls["n"] += 1
        raise urllib.error.URLError("down")

    monkeypatch.setattr(urllib.request, "urlopen", always_fail)
    model = OpenAICompatChatModel(
        "http://example.invalid/v1",
        "m",
        api_key="k",
        max_attempts=2,
        retry_backoff_s=0.0,
        timeout_s=1,
    )
    with pytest.raises(GenerationError):
        model.chat([{"role": "user", "content": "hi"}])
    assert calls["n"] == 2


def test_openai_compat_does_not_retry_a_bad_request(monkeypatch):
    """Retrying a 401 just burns the caller's quota; fail fast instead."""
    import io
    import urllib.error
    import urllib.request

    from issue_rag.generate import GenerationError, OpenAICompatChatModel

    calls = {"n": 0}

    def unauthorized(req, timeout=None):
        calls["n"] += 1
        raise urllib.error.HTTPError(
            "http://example.invalid/v1/chat/completions",
            401,
            "Unauthorized",
            {},
            io.BytesIO(b'{"error":"bad key"}'),
        )

    monkeypatch.setattr(urllib.request, "urlopen", unauthorized)
    model = OpenAICompatChatModel(
        "http://example.invalid/v1",
        "m",
        api_key="k",
        max_attempts=3,
        retry_backoff_s=0.0,
        timeout_s=1,
    )
    with pytest.raises(GenerationError):
        model.chat([{"role": "user", "content": "hi"}])
    assert calls["n"] == 1, "4xx must not be retried"


def test_openai_compat_retries_tls_truncation(monkeypatch):
    """HTTPS here goes through an inspection proxy that drops connections.

    The failure surfaces as ``[SSL: UNEXPECTED_EOF_WHILE_READING]``, either
    wrapped in ``URLError`` or raised raw from the ssl layer. Both shapes are
    transient and must be retried; treating them as fatal is what left 12 of 95
    closed-book items with no answer last run.
    """
    import ssl
    import urllib.error
    import urllib.request

    from issue_rag.generate import OpenAICompatChatModel

    calls = {"n": 0}

    def flaky(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.URLError(ssl.SSLEOFError(8, "EOF in violation of protocol"))
        if calls["n"] == 2:
            raise ssl.SSLEOFError(8, "EOF occurred in violation of protocol")
        return _FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", flaky)
    model = OpenAICompatChatModel(
        "http://example.invalid/v1",
        "m",
        api_key="k",
        max_attempts=5,
        retry_backoff_s=0.0,
        timeout_s=1,
    )
    result = model.chat([{"role": "user", "content": "hi"}])

    assert result.text == "the answer"
    assert calls["n"] == 3, "both TLS-truncation shapes must be retried"


def test_summarise_excludes_errors_from_accuracy_and_reports_them():
    from issue_rag.evaluate import summarise

    records = [
        {"verdict": "correct", "score": 1.0, "fabricated": False, "cited": True, "latency_s": 1.0},
        {"verdict": "incorrect", "score": 0.0, "fabricated": True, "cited": True, "latency_s": 2.0},
        {"verdict": "error", "score": 0.0, "fabricated": False, "cited": False, "latency_s": 0.5, "error": "boom"},
    ]
    row = summarise(records, "demo")
    assert row["n"] == 2, "errors must not count as evaluated items"
    assert row["errors"] == 1
    assert row["error_rate"] == 33.3
    assert row["accuracy"] == 0.5


# --------------------------------------------------- grader-sensitivity harness


def test_failed_call_is_not_treated_as_done(monkeypatch):
    """A recorded failure is not an answer, so resume must rerun it.

    Regression: ``load_shard`` keyed by item id and ignored the ``error`` field,
    so a failed generation was cached as done. A 95-item run then reported
    ``n=83, errors=12`` and nothing in the report said "12 items are missing".
    """
    import argparse
    import shutil

    from issue_rag import evaluate as ev
    from issue_rag.config import RepoPreset
    from issue_rag.generate import GenerationResult

    work = Path(__file__).resolve().parent / "_shard_tmp"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(ev, "RESULTS_DIR", work)

    preset = RepoPreset(key="vite", repo="vitejs/vite", display_name="Vite")
    items = [
        {"id": "q1", "query": "how do I do x", "reference_answer": "do y"},
        {"id": "q2", "query": "why is z slow", "reference_answer": "because w"},
    ]
    shard = ev.shard_path(preset, "closed_book")
    shard.write_text(
        json.dumps(
            {
                "item_id": "q1",
                "system": "closed_book",
                "answer": "y",
                "verdict": "correct",
                "score": 1.0,
                "fabricated": False,
                "cited": True,
                "latency_s": 1.0,
                "generated_tokens": 3,
            }
        )
        + "\n"
        + json.dumps(
            {
                "item_id": "q2",
                "system": "closed_book",
                "answer": "",
                "verdict": "error",
                "score": 0.0,
                "error": "generation failed after 3 attempts: SSL EOF",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    asked: list[str] = []

    class FakeModel:
        name = "fake"

        def chat(self, messages, *, max_new_tokens=512, temperature=0.0):
            asked.append(messages[-1]["content"])
            return GenerationResult(
                text="recovered answer", prompt_tokens=1, output_tokens=1, latency_s=0.0
            )

    class FakeJudge:
        class model:  # noqa: N801 - mirrors the real attribute shape
            name = "fake"

        def grade(self, query, reference, candidate):
            return type(
                "G",
                (),
                {
                    "verdict": "partial",
                    "score": 0.5,
                    "fabricated": False,
                    "checks": {},
                    "reason": "ok",
                },
            )()

    args = argparse.Namespace(
        run_tag="", top_k=5, max_new_tokens=64, dense_model=None, reranker_model=None, device="cpu"
    )
    try:
        ev.run_system(
            "closed_book", items, None, model=FakeModel(), judge=FakeJudge(), preset=preset, args=args
        )

        assert asked == ["why is z slow"], "only the failed item may be regenerated"
        records = ev.load_shard(shard)
        assert ev.finished_ids(records) == {"q1", "q2"}
        assert records["q2"]["answer"] == "recovered answer"
        assert "error" not in records["q2"], "the success must supersede the failure"
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_unparsed_grade_is_not_treated_as_done():
    """Same rule as the generation path: no parseable verdict means retry."""
    from issue_rag.evaluate import finished_ids

    records = {
        "q1": {"item_id": "q1", "verdict": "correct", "score": 1.0},
        "q2": {"item_id": "q2", "verdict": "unparsed", "score": 0.0},
        "q3": {"item_id": "q3", "verdict": "error", "score": 0.0, "error": "ssl"},
    }
    assert finished_ids(records) == {"q1"}


def test_shard_path_is_tag_scoped():
    """Two judges must not share a cache key, or the second run is skipped."""
    from issue_rag.config import RepoPreset
    from issue_rag.evaluate import results_stem, shard_path

    preset = RepoPreset(key="vite", repo="vitejs/vite", display_name="Vite")
    base = shard_path(preset, "bm25")
    tagged = shard_path(preset, "bm25", "reasoner")
    assert base.name == "vitejs__vite__bm25.jsonl"
    assert tagged.name == "vitejs__vite__bm25__reasoner.jsonl"
    assert results_stem(preset) == "vitejs__vite__results"
    assert results_stem(preset, "reasoner") == "vitejs__vite__reasoner__results"


def test_compare_records_only_counts_items_present_in_both():
    from issue_rag.sensitivity import compare_records

    baseline = {
        "a": {"item_id": "a", "verdict": "correct"},
        "b": {"item_id": "b", "verdict": "incorrect"},
        "c": {"item_id": "c", "verdict": "error", "error": "boom"},
    }
    variant = {
        "a": {"item_id": "a", "verdict": "correct"},
        "b": {"item_id": "b", "verdict": "partial"},
        "d": {"item_id": "d", "verdict": "correct"},
    }
    row = compare_records(baseline, variant)
    assert row["n"] == 2, "errors and one-sided items must be dropped"
    assert row["baseline_accuracy"] == 0.5
    assert row["variant_accuracy"] == 0.75
    assert row["exact_agreement"] == 50.0
    # incorrect -> partial flips usability, so the BINARY reading disagrees even
    # though the answer did not get worse -- this is the boundary the two graders
    # actually fight over.
    assert row["binary_agreement"] == 50.0
    assert row["verdict_transitions"] == {"incorrect -> partial": 1}


def test_compare_records_reports_no_overlap():
    from issue_rag.sensitivity import compare_records

    assert compare_records({"a": {"verdict": "correct"}}, {"b": {"verdict": "correct"}}) == {
        "n": 0
    }


def test_regrade_without_a_run_tag_is_refused():
    """Appending to the shards it is reading from would corrupt the baseline."""
    from issue_rag.evaluate import main

    with pytest.raises(SystemExit):
        main(["--preset", "vite", "--regrade", "--judge-model", "x"])


# ------------------------------------------------------- judge: operational rubric


class _StubJudgeModel:
    """Returns a fixed completion; lets us test the judge without a network."""

    name = "stub"

    def __init__(self, payload: str) -> None:
        self.payload = payload

    def chat(self, messages, *, max_new_tokens=512, temperature=0.0):
        from issue_rag.generate import GenerationResult

        return GenerationResult(text=self.payload)


def _checks(**overrides) -> dict:
    base = {
        "states_resolution": False,
        "generic_advice_only": False,
        "wrong_specific": False,
        "contradicts": False,
    }
    base.update(overrides)
    return base


def test_verdict_mapping_is_pinned():
    """The mapping is the part that used to drift between runs; pin every row."""
    from issue_rag.judge import verdict_from_checks

    assert verdict_from_checks(_checks(contradicts=True)) == "incorrect"
    assert verdict_from_checks(_checks(states_resolution=True)) == "correct"
    assert (
        verdict_from_checks(_checks(states_resolution=True, wrong_specific=True))
        == "partial"
    )
    assert verdict_from_checks(_checks(generic_advice_only=True)) == "incorrect"
    assert verdict_from_checks(_checks(wrong_specific=True)) == "incorrect"
    assert verdict_from_checks(_checks()) == "partial"


def test_contradiction_dominates_every_other_flag():
    from issue_rag.judge import verdict_from_checks

    for states in (True, False):
        for generic in (True, False):
            for fab in (True, False):
                checks = _checks(
                    contradicts=True,
                    states_resolution=states,
                    generic_advice_only=generic,
                    wrong_specific=fab,
                )
                assert verdict_from_checks(checks) == "incorrect"


def test_verdict_mapping_covers_all_sixteen_combinations():
    import itertools

    from issue_rag.judge import VERDICTS, verdict_from_checks

    for combo in itertools.product([True, False], repeat=4):
        checks = dict(zip(("states_resolution", "generic_advice_only",
                           "wrong_specific", "contradicts"), combo))
        assert verdict_from_checks(checks) in VERDICTS


def test_extract_checks_rejects_missing_keys():
    """A missing boolean must not silently default to False and earn credit."""
    from issue_rag.judge import extract_checks

    assert extract_checks({"states_resolution": True}) is None
    partial_payload = _checks()
    del partial_payload["wrong_specific"]
    assert extract_checks(partial_payload) is None


def test_extract_checks_accepts_strings_and_ints():
    from issue_rag.judge import extract_checks

    parsed = {
        "states_resolution": "true",
        "generic_advice_only": "false",
        "wrong_specific": 1,
        "contradicts": 0,
    }
    assert extract_checks(parsed) == {
        "states_resolution": True,
        "generic_advice_only": False,
        "wrong_specific": True,
        "contradicts": False,
    }


def test_judge_grades_through_the_mapping_not_the_model():
    """The model only answers sub-questions; the verdict comes from code."""
    from issue_rag.judge import Judge

    payload = json.dumps(
        _checks(states_resolution=True, wrong_specific=True)
        | {"reason": "right fix, invented version"}
    )
    grade = Judge(_StubJudgeModel(payload)).grade("q", "ref", "cand")
    assert grade.verdict == "partial", "fabrication caps a correct fix at partial"
    assert grade.fabricated is True
    assert grade.checks["states_resolution"] is True


def test_judge_marks_legacy_output_distinctly():
    """A stale prompt emitting a bare verdict must be visible, not silently equal."""
    from issue_rag.judge import Judge

    payload = json.dumps({"verdict": "correct", "fabricated": False})
    grade = Judge(_StubJudgeModel(payload)).grade("q", "ref", "cand")
    assert grade.verdict == "correct"
    assert grade.meta.get("legacy_output") is True
    assert grade.checks == {}


def test_judge_returns_unparsed_on_garbage():
    from issue_rag.judge import Judge

    grade = Judge(_StubJudgeModel("no json here")).grade("q", "ref", "cand")
    assert grade.verdict == "unparsed"
    assert grade.score == 0.0


def test_judge_empty_candidate_is_incorrect_without_a_call():
    from issue_rag.judge import Judge

    stub = _StubJudgeModel("should not be used")
    grade = Judge(stub).grade("q", "ref", "   ")
    assert grade.verdict == "incorrect"
    assert grade.raw == ""


def test_regrade_refuses_to_write_into_the_arm_it_reads():
    """Reading arm X and writing arm X would double-count and corrupt X."""
    from issue_rag.evaluate import main

    with pytest.raises(SystemExit):
        main(["--preset", "vite", "--regrade", "--regrade-from", "v1holistic",
              "--run-tag", "v1holistic", "--judge-model", "x"])


# ------------------------------------------------------------- human review tool


def _review_entry(**overrides):
    base = {
        "review_id": "closed_book:q1",
        "item_id": "q1",
        "system": "closed_book",
        "question": "why does the build hang",
        "reference": "Fixed in vite 8.0.6 by upgrading rolldown.",
        "candidate": "Try clearing your cache and updating Node.",
        "judge_verdict": "incorrect",
        "judge_checks": {"generic_advice_only": True},
    }
    base.update(overrides)
    return base


def test_render_hides_the_judge_verdict():
    """A review is only blind if the verdict is not on screen."""
    from issue_rag.human_review import render

    text = render(_review_entry(), 1, 30, 80)
    assert "incorrect" not in text
    assert "judge" not in text.lower()


def test_render_hides_which_system_produced_the_answer():
    """Knowing it came from closed_book biases a reviewer toward harshness."""
    from issue_rag.human_review import render

    text = render(_review_entry(), 1, 30, 80)
    assert "closed_book" not in text


def test_render_shows_question_reference_and_candidate():
    from issue_rag.human_review import render

    text = render(_review_entry(), 1, 30, 80)
    assert "why does the build hang" in text
    assert "Fixed in vite 8.0.6" in text
    assert "Try clearing your cache" in text
    assert "[1/30]" in text


def test_weighted_kappa_is_one_under_perfect_agreement():
    from issue_rag.human_review import quadratically_weighted_kappa

    labels = ["correct", "partial", "incorrect", "correct", "partial"]
    assert abs(quadratically_weighted_kappa(labels, labels) - 1.0) < 1e-9


def test_weighted_kappa_penalises_one_step_less_than_two():
    """Ordinal scales: correct-vs-partial is a smaller error than correct-vs-incorrect.

    The two comparisons must keep the marginals identical, or kappa compares
    different chance levels instead of different penalties. Swapping two label
    positions does that, which is why the judge shown here is a permutation of the
    human labels rather than a uniform shift.
    """
    from issue_rag.human_review import quadratically_weighted_kappa

    human = ["correct", "partial", "incorrect", "correct", "partial", "incorrect"]
    one_step = ["partial", "correct", "incorrect", "correct", "partial", "incorrect"]
    two_step = ["incorrect", "partial", "correct", "correct", "partial", "incorrect"]
    assert quadratically_weighted_kappa(human, one_step) > quadratically_weighted_kappa(
        human, two_step
    )


def test_weighted_kappa_is_negative_when_agreement_is_worse_than_chance():
    from issue_rag.human_review import quadratically_weighted_kappa

    human = ["correct"] * 5 + ["incorrect"] * 5
    judge = ["incorrect"] * 5 + ["correct"] * 5
    assert quadratically_weighted_kappa(human, judge) < 0


def test_weighted_kappa_handles_empty_input():
    from issue_rag.human_review import quadratically_weighted_kappa

    assert quadratically_weighted_kappa([], []) is None


def test_retrieval_metrics_score_the_reference_comment_not_the_issue():
    """The issue-level recall is a tautology on this corpus; the gold chunk is not.

    Two items: on the first the answering comment is retrieved at rank 2, on the
    second the window contains only *other* comments from the same issue. Both
    count as an issue-level hit (that is the tautology), only the first counts as
    a gold-chunk hit.
    """
    from types import SimpleNamespace

    from issue_rag.evaluate import evaluate_retrieval

    def hit(issue_number, comment_id=None):
        return SimpleNamespace(
            issue_number=issue_number,
            doc_id=f"issue-{issue_number}-comment-{comment_id}",
            metadata={"comment_id": comment_id},
        )

    class FakeRetriever:
        def search(self, query, top_k=5, exclude_issue=None):
            if query == "q1":
                window = [hit(7, "111"), hit(7, "999")]
            else:
                window = [hit(8, "222"), hit(8, "333")]
            return [h for h in window if h.issue_number != exclude_issue]

    items = [
        {"query": "q1", "issue_number": 7, "reference_comment_ids": [999]},
        {"query": "q2", "issue_number": 8, "reference_comment_ids": [444]},
    ]
    metrics = evaluate_retrieval(FakeRetriever(), items, top_k=5)

    assert metrics["recall@5"] == 1.0, "both items retrieve their own issue"
    assert metrics["recall@5_loo"] == 0.0, "structurally zero, by definition"
    assert metrics["gold_chunk@5"] == 0.5, "only q1 retrieves the answering comment"
    assert metrics["gold_chunk_top1"] == 0.0, "it was at rank 2, not rank 1"
    assert metrics["gold_chunk_mrr"] == 0.25, "1/2 for q1, 0 for q2, averaged"
    assert metrics["self_share@5"] == 1.0, "the whole window comes from the same issue"


def test_demo_page_renders_answer_markdown_instead_of_escaping_it():
    """The answer used to be shown with ``white-space: pre-wrap`` and HTML-escaped,
    so readers saw literal asterisks and backticks: "too messy to read". The
    renderer still escapes first -- model output is untrusted."""
    from issue_rag.serve import _inline_md

    rendered = _inline_md("Vite defers `.env` loading.\n\n**Because** of `root`.")
    assert "<code>.env</code>" in rendered
    assert "<strong>Because</strong>" in rendered
    assert rendered.count("<p>") == 2, "blank line should start a new paragraph"

    xss = _inline_md("<script>alert(1)</script>")
    assert "<script>" not in xss and "&lt;script&gt;" in xss


def test_demo_page_citations_link_to_the_source_cards():
    """``[3]`` was plain text, so a reader had to count cards to find the source."""
    from issue_rag.serve import _cite_links

    linked = _cite_links("see [1] and [3] and [99]", 5)
    assert 'href="#src-1"' in linked
    assert 'href="#src-3"' in linked
    assert "[99]" in linked, "out-of-range citations must be left alone"


def test_demo_page_strips_the_duplicated_title_prefix():
    from issue_rag.serve import _strip_title_prefix

    chunk = "Issue #23588: New Vite SVG logo renders incorrectly\n\nok, I reproduced it"
    assert _strip_title_prefix(chunk) == "ok, I reproduced it"
    assert _strip_title_prefix("no prefix here") == "no prefix here"


def test_review_page_storage_key_changes_with_the_queue():
    """A new batch must not inherit the previous batch's browser progress.

    Regression: the page kept decisions under a constant localStorage key while
    addressing entries by bare index, so opening a rebuilt queue showed "all 30
    already graded" from the previous session's review.
    """
    from issue_rag.human_review import _page_for

    def entry(review_id, item_id):
        return {
            "review_id": review_id,
            "item_id": item_id,
            "question": "q",
            "reference": "r",
            "candidate": "c",
        }

    first = _page_for([entry("closed_book:q1", "q1")])
    replay = _page_for([entry("closed_book:q1", "q1")])
    second = _page_for([entry("hybrid_rerank:q2", "q2")])

    assert first == replay, "same queue must keep the same key, or progress is lost"
    assert first != second, "a different queue must get a different key"

    import re

    keys = re.findall(r'localStorage\.getItem\("([^"]+)"\)', first)
    assert len(keys) == 1
    assert keys[0].startswith("issueRagDecisions:")
    assert "closed_book" not in keys[0], "the key must not name a system"


def test_review_page_cannot_leak_the_judge_verdict():
    """The embedded JSON must carry no per-item verdict, checks, or system name.

    A string search for "incorrect" is useless here: the page legitimately contains
    all three verdict words as the reviewer's own option values. What matters is
    that no queue entry carries the judge's answer.
    """
    import re

    from issue_rag.human_review import _page_for

    page = _page_for([_review_entry()])
    match = re.search(r"/\*QUEUE_START\*/(.*?)/\*QUEUE_END\*/", page, re.S)
    assert match, "the page should embed the queue as JSON"
    entries = json.loads(match.group(1))
    assert len(entries) == 1
    allowed = {"key", "item_id", "question", "reference", "candidate"}
    assert set(entries[0]) == allowed
    assert not any("judge" in name for name in entries[0])
    assert "judge_verdict" not in page
    assert "judge_checks" not in page
    # review_id embeds the system name ("closed_book:q1"), so the page addresses
    # entries by index; otherwise the page source reveals who answered.
    assert "closed_book" not in page, "the system name must not be on the page"
    assert entries[0]["key"] == 0


def test_review_page_contains_the_material_to_judge():
    import re

    from issue_rag.human_review import _page_for

    page = _page_for([_review_entry(), _review_entry(review_id="b:q2", item_id="q2")])
    assert "why does the build hang" in page
    assert "Fixed in vite 8.0.6" in page
    assert "Try clearing your cache" in page
    match = re.search(r"/\*QUEUE_START\*/(.*?)/\*QUEUE_END\*/", page, re.S)
    entries = json.loads(match.group(1))
    assert len(entries) == 2, "every queued review must be addressable"
    assert [entry["key"] for entry in entries] == [0, 1]
