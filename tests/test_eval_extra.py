"""Additional tests for eval module — covers LLM judge and edge cases."""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

from hybrid_rag.eval import (
    dcg_at_k,
    lexical_faithfulness,
    llm_judge_faithfulness,
    log2,
    mean,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestLog2:
    def test_log2_one(self):
        assert log2(1.0) == 0.0

    def test_log2_two(self):
        assert log2(2.0) == 1.0

    def test_log2_four(self):
        assert log2(4.0) == 2.0


class TestMean:
    def test_mean_normal(self):
        assert mean([1.0, 2.0, 3.0]) == 2.0

    def test_mean_empty(self):
        assert mean([]) == 0.0

    def test_mean_single(self):
        assert mean([5.0]) == 5.0


class TestRecallAtK:
    def test_empty_relevant(self):
        assert recall_at_k(["d1", "d2"], set(), 5) == 0.0

    def test_all_relevant(self):
        assert recall_at_k(["d1", "d2"], {"d1", "d2"}, 5) == 1.0

    def test_partial(self):
        assert recall_at_k(["d1", "d3"], {"d1", "d2"}, 5) == 0.5


class TestPrecisionAtK:
    def test_empty_k(self):
        assert precision_at_k(["d1"], {"d1"}, 0) == 0.0

    def test_perfect(self):
        assert precision_at_k(["d1", "d2"], {"d1", "d2"}, 2) == 1.0

    def test_partial(self):
        assert precision_at_k(["d1", "d3"], {"d1", "d2"}, 2) == 0.5


class TestReciprocalRank:
    def test_first_hit(self):
        assert reciprocal_rank(["d1"], {"d1"}) == 1.0

    def test_second_hit(self):
        assert reciprocal_rank(["d2", "d1"], {"d1"}) == 0.5

    def test_no_hit(self):
        assert reciprocal_rank(["d3", "d4"], {"d1"}) == 0.0


class TestDCG:
    def test_dcg_known(self):
        result = dcg_at_k(["d1", "d2", "d3"], {"d1", "d3"}, 3)
        expected = 1.0 / log2(2) + 0 + 1.0 / log2(4)
        assert abs(result - expected) < 1e-6

    def test_dcg_empty(self):
        assert dcg_at_k([], set(), 5) == 0.0


class TestNDCG:
    def test_perfect_ndcg(self):
        result = ndcg_at_k(["d1", "d2"], {"d1", "d2"}, 2)
        assert abs(result - 1.0) < 1e-6

    def test_imperfect_ndcg(self):
        # When relevant docs are not in top positions, NDCG < 1
        result = ndcg_at_k(["d3", "d4", "d1"], {"d1", "d2"}, 3)
        assert result < 1.0

    def test_ndcg_no_relevant(self):
        assert ndcg_at_k(["d1"], set(), 5) == 0.0


class TestLexicalFaithfulness:
    def test_identical_tokens(self):
        assert lexical_faithfulness("hello world", "hello world") == 1.0

    def test_no_overlap(self):
        assert lexical_faithfulness("cat dog", "bird fish") == 0.0

    def test_partial_overlap(self):
        result = lexical_faithfulness("hello world test", "hello earth")
        assert 0.0 < result < 1.0

    def test_empty_answer(self):
        assert lexical_faithfulness("", "source") == 0.0

    def test_empty_source(self):
        assert lexical_faithfulness("answer", "") == 0.0

    def test_both_empty(self):
        assert lexical_faithfulness("", "") == 0.0


class TestLLMJudgeFaithfulness:
    def test_no_api_key_returns_none(self):
        with patch.dict(os.environ, {}, clear=True):
            assert llm_judge_faithfulness("q", "a", "s") is None

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_returns_score(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="4"))]
        mock_client.chat.completions.create.return_value = mock_resp

        result = llm_judge_faithfulness("query", "answer", "source")
        assert result == 4

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_clamps_score_min(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="0"))]
        mock_client.chat.completions.create.return_value = mock_resp

        result = llm_judge_faithfulness("q", "a", "s")
        assert result == 1

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_clamps_score_max(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="10"))]
        mock_client.chat.completions.create.return_value = mock_resp

        result = llm_judge_faithfulness("q", "a", "s")
        assert result == 5

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_none_content_returns_none(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content=None))]
        mock_client.chat.completions.create.return_value = mock_resp

        result = llm_judge_faithfulness("q", "a", "s")
        assert result is None

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_exception_returns_none(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API error")

        result = llm_judge_faithfulness("q", "a", "s")
        assert result is None

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_strips_whitespace(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="  3  "))]
        mock_client.chat.completions.create.return_value = mock_resp

        result = llm_judge_faithfulness("q", "a", "s")
        assert result == 3
