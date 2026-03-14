"""Unit tests for local mode functions (no API key or server needed)."""

from unittest.mock import patch

from jina_cli.api import local_classify, _cosine_similarity


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
