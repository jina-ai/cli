"""Unit tests for local mode functions (no API key or server needed)."""

from unittest.mock import patch

from click.testing import CliRunner

from jina_cli.api import local_classify, _cosine_similarity
from jina_cli.main import cli


class TestCosineSimlarity:
    def test_identical_vectors(self):
        assert abs(_cosine_similarity([1, 0, 0], [1, 0, 0]) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        assert abs(_cosine_similarity([1, 0], [0, 1])) < 1e-6

    def test_zero_vector(self):
        assert _cosine_similarity([0, 0], [1, 1]) == 0.0


class TestLocalClassify:
    def test_single_text(self):
        fake_embeddings = [
            {"embedding": [0.9, 0.1, 0.0]},   # "I love this" - text
            {"embedding": [0.8, 0.2, 0.0]},   # "positive" - label (close)
            {"embedding": [0.0, 0.1, 0.9]},   # "negative" - label (far)
        ]

        with patch("jina_cli.api.local_embed", return_value=fake_embeddings):
            result = local_classify(
                texts=["I love this"],
                labels=["positive", "negative"],
            )

        assert len(result) == 1
        assert result[0]["prediction"] == "positive"
        assert result[0]["score"] > 0.5
        assert result[0]["index"] == 0

    def test_multiple_texts(self):
        fake_embeddings = [
            {"embedding": [0.9, 0.1]},   # text 1 - closer to label 1
            {"embedding": [0.1, 0.9]},   # text 2 - closer to label 2
            {"embedding": [0.8, 0.2]},   # label "sports"
            {"embedding": [0.2, 0.8]},   # label "politics"
        ]

        with patch("jina_cli.api.local_embed", return_value=fake_embeddings):
            result = local_classify(
                texts=["goal scored", "election results"],
                labels=["sports", "politics"],
            )

        assert len(result) == 2
        assert result[0]["prediction"] == "sports"
        assert result[1]["prediction"] == "politics"

    def test_result_format(self):
        """Results should have index, prediction, score keys."""
        fake_embeddings = [
            {"embedding": [1.0, 0.0]},
            {"embedding": [0.9, 0.1]},
        ]

        with patch("jina_cli.api.local_embed", return_value=fake_embeddings):
            result = local_classify(
                texts=["test"],
                labels=["label1"],
            )

        assert "index" in result[0]
        assert "prediction" in result[0]
        assert "score" in result[0]
        assert result[0]["prediction"] == "label1"


class TestDedupKValidation:
    """Non-positive -k must be rejected before any API/local call (offline)."""

    def test_dedup_zero_k_rejected(self, monkeypatch):
        monkeypatch.delenv("JINA_API_KEY", raising=False)
        result = CliRunner().invoke(
            cli, ["dedup", "-k", "0"], input="apple\nbanana\ncherry\n"
        )
        assert result.exit_code == 1
        assert "-k must be a positive integer" in result.stderr
        assert "API key" not in result.stderr

    def test_dedup_negative_k_rejected(self, monkeypatch):
        monkeypatch.delenv("JINA_API_KEY", raising=False)
        result = CliRunner().invoke(
            cli, ["dedup", "-k", "-2"], input="apple\nbanana\n"
        )
        assert result.exit_code == 1
        assert "-k must be a positive integer" in result.stderr

    def test_dedup_valid_k_passes_validation(self, monkeypatch):
        """A valid -k proceeds past validation (fails later on API key)."""
        monkeypatch.delenv("JINA_API_KEY", raising=False)
        result = CliRunner().invoke(
            cli, ["dedup", "-k", "2"], input="apple\nbanana\ncherry\n"
        )
        assert "-k must be a positive integer" not in result.stderr
