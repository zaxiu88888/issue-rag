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
