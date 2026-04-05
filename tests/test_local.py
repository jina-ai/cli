"""Unit tests for local mode functions (no API key or server needed)."""

import subprocess
import sys
from unittest.mock import MagicMock, patch

from jina_cli.api import local_classify, _cosine_similarity, DEFAULT_TIMEOUT


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


def run_jina(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "jina_cli.main", *args],
        capture_output=True,
        text=True,
        timeout=10,
        input=stdin,
        env={"PATH": "/usr/bin:/bin"},
    )


class TestTimeoutFlag:
    def test_global_timeout_in_help(self):
        """--timeout should appear in global --help."""
        r = subprocess.run(
            [sys.executable, "-m", "jina_cli.main", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert "--timeout" in r.stdout or "--timeout" in r.stderr

    def test_subcommand_timeout_in_help(self):
        """--timeout should appear in subcommand --help."""
        for cmd in ["read", "search", "embed", "rerank", "dedup", "classify",
                    "screenshot", "bibtex", "expand", "pdf", "datetime", "primer"]:
            r = subprocess.run(
                [sys.executable, "-m", "jina_cli.main", cmd, "--help"],
                capture_output=True, text=True, timeout=10,
            )
            assert "--timeout" in r.stdout or "--timeout" in r.stderr, \
                f"--timeout missing from `jina {cmd} --help`"

    def test_api_functions_accept_timeout(self):
        """API functions should pass timeout to httpx client."""
        import jina_cli.api as api_module

        mock_resp = MagicMock()
        mock_resp.text = "content"
        mock_resp.json.return_value = {"data": {}}
        mock_resp.status_code = 200

        with patch("jina_cli.api._client") as mock_client_factory:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_resp
            mock_resp.raise_for_status = MagicMock()
            mock_client_factory.return_value = mock_client

            api_module.read_url("https://example.com", timeout=60.0)
            mock_client_factory.assert_called_with(timeout=60.0)

    def test_default_timeout_unchanged(self):
        """DEFAULT_TIMEOUT should remain 30.0."""
        assert DEFAULT_TIMEOUT == 30.0

    def test_global_timeout_propagates(self):
        """--timeout passed globally should be stored in context."""
        import click
        from click.testing import CliRunner
        from jina_cli.main import cli

        runner = CliRunner()

        captured = {}

        @cli.command("_test_timeout")
        @click.option("--timeout", default=None, type=float)
        @click.pass_context
        def _test_cmd(ctx, timeout):
            captured["timeout"] = ctx.obj.get("timeout")

        result = runner.invoke(cli, ["--timeout", "99.0", "_test_timeout"])
        assert captured.get("timeout") == 99.0
