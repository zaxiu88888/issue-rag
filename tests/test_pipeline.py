"""Unit tests for the parts that can run with no network, GPU, or model weights.

The heavy stages (harvest, embed, generate) are not covered here on purpose:
they are expensive and flaky to test hermetically. What *is* covered is every
piece of logic that could silently corrupt the experiment -- chunking, the BM25
tokenizer, rank fusion, eval-set filtering, and judge output parsing.

CI runs exactly this file, which is why it must stay dependency-light.
"""

from __future__ import annotations

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
