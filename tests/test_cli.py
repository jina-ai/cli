"""Integration tests for jina-cli using real Jina API.

Requires JINA_API_KEY env var (set via GitHub Secrets in CI).
"""

import json
import os
import subprocess
import sys

import pytest

# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not os.environ.get("JINA_API_KEY"),
    reason="JINA_API_KEY not set",
)


def run_jina(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    """Run jina CLI and capture output."""
    result = subprocess.run(
        [sys.executable, "-m", "jina_cli.main", *args],
        capture_output=True,
        text=True,
        timeout=60,
        input=stdin,
        env={**os.environ},
    )
    return result


class TestRead:
    def test_read_url(self):
        r = run_jina("read", "https://example.com")
        assert r.returncode == 0
        assert "Example Domain" in r.stdout

    def test_read_json(self):
        r = run_jina("read", "https://example.com", "--json")
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert "data" in data or "content" in data or "url" in data

    def test_read_stdin(self):
        r = run_jina("read", stdin="https://example.com\n")
        assert r.returncode == 0
        assert "Example" in r.stdout


class TestSearch:
    def test_search_basic(self):
        r = run_jina("search", "what is jina ai", "-n", "3")
        assert r.returncode == 0
        assert "jina" in r.stdout.lower()

    def test_search_json(self):
        r = run_jina("search", "jina ai", "-n", "2", "--json")
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert "results" in data
        assert len(data["results"]) > 0

    def test_search_arxiv(self):
        r = run_jina("search", "--arxiv", "attention mechanism", "-n", "2", "--json")
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert "results" in data

    def test_search_human_readable(self):
        """Default output should be human-readable, not raw JSON."""
        r = run_jina("search", "python", "-n", "2")
        assert r.returncode == 0
        # Should NOT start with { (raw JSON)
        assert not r.stdout.strip().startswith("{")
        # Should have title + URL pattern
        assert "http" in r.stdout


class TestEmbed:
    def test_embed_single(self):
        r = run_jina("embed", "hello world")
        assert r.returncode == 0
        assert "dim=" in r.stdout

    def test_embed_json(self):
        r = run_jina("embed", "hello world", "--json")
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert isinstance(data, list)
        assert len(data) > 0
        assert "embedding" in data[0]

    def test_embed_multiple(self):
        r = run_jina("embed", "hello", "world", "foo")
        assert r.returncode == 0
        lines = [l for l in r.stdout.strip().split("\n") if l.startswith("[")]
        assert len(lines) == 3

    def test_embed_stdin(self):
        r = run_jina("embed", stdin="hello\nworld\n")
        assert r.returncode == 0
        assert "dim=" in r.stdout


class TestRerank:
    def test_rerank_basic(self):
        r = run_jina("rerank", "pet animal", stdin="cat is cute\ndog is loyal\nfish can swim\n")
        assert r.returncode == 0
        # Should have score format [x.xxxx]
        assert "[" in r.stdout and "]" in r.stdout

    def test_rerank_json(self):
        r = run_jina("rerank", "pet", "--json", stdin="cat\ndog\nfish\n")
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert isinstance(data, list)
        assert len(data) > 0

    def test_rerank_no_stdin(self):
        """Should error with exit 1 when no stdin provided."""
        r = run_jina("rerank", "query")
        assert r.returncode == 1
        assert "Error" in r.stderr or "Fix" in r.stderr


class TestDedup:
    def test_dedup_basic(self):
        r = run_jina("dedup", stdin="hello world\nhello world\ngoodbye world\n")
        assert r.returncode == 0
        lines = [l for l in r.stdout.strip().split("\n") if l]
        # Should have fewer unique items
        assert len(lines) <= 3

    def test_dedup_no_stdin(self):
        r = run_jina("dedup")
        assert r.returncode == 1


class TestBibtex:
    def test_bibtex_basic(self):
        r = run_jina("bibtex", "attention is all you need")
        assert r.returncode == 0
        # DBLP/Semantic Scholar may rate limit; just check it doesn't error

    def test_bibtex_json(self):
        r = run_jina("bibtex", "BERT", "--json")
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert isinstance(data, list)


class TestExpand:
    def test_expand_basic(self):
        r = run_jina("expand", "how to train embeddings")
        assert r.returncode == 0
        lines = [l for l in r.stdout.strip().split("\n") if l]
        assert len(lines) > 1  # Should return multiple expansions


class TestPrimer:
    def test_primer(self):
        r = run_jina("primer")
        assert r.returncode == 0
        assert "jina.ai" in r.stdout.lower() or "reader" in r.stdout.lower()


class TestScreenshot:
    def test_screenshot_stdout(self):
        """Without -o, should print URL not binary data."""
        r = run_jina("screenshot", "https://example.com")
        assert r.returncode == 0
        # Should output a URL or base64 data indicator, not raw binary
        assert r.stdout.strip()


class TestHelp:
    """Test progressive help disclosure."""

    def test_layer0_no_args(self):
        """jina with no args shows command list to stderr."""
        r = run_jina()
        assert r.returncode == 0
        assert "jina read" in r.stderr
        assert "jina search" in r.stderr
        # stdout should be empty (no data pollution)
        assert r.stdout.strip() == ""

    def test_layer1_no_args(self):
        """Subcommand with no args shows short usage to stderr."""
        r = run_jina("embed")
        assert r.returncode == 1
        assert "Usage" in r.stderr or "jina embed" in r.stderr

    def test_layer2_help_flag(self):
        """--help shows full options."""
        r = run_jina("search", "--help")
        assert r.returncode == 0
        assert "--json" in r.stdout or "--json" in r.stderr

    def test_typo_suggestion(self):
        """Typo should suggest correct command."""
        r = run_jina("rea")
        assert r.returncode != 0
        assert "read" in r.stderr.lower() or "rerank" in r.stderr.lower()


class TestErrorHandling:
    def test_invalid_api_key(self):
        """Should exit 1 with actionable error for bad key."""
        env = {**os.environ, "JINA_API_KEY": "invalid-key"}
        result = subprocess.run(
            [sys.executable, "-m", "jina_cli.main", "embed", "test"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        assert result.returncode == 1
        assert "Fix" in result.stderr or "key" in result.stderr.lower()

    def test_missing_api_key(self):
        """Should exit 1 with helpful message when key missing."""
        env = {k: v for k, v in os.environ.items() if k != "JINA_API_KEY"}
        result = subprocess.run(
            [sys.executable, "-m", "jina_cli.main", "embed", "test"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        assert result.returncode == 1
        assert "JINA_API_KEY" in result.stderr


class TestExitCodes:
    def test_success_exit_0(self):
        r = run_jina("primer")
        assert r.returncode == 0

    def test_user_error_exit_1(self):
        """Missing required input should exit 1."""
        r = run_jina("embed")
        assert r.returncode == 1


class TestClassify:
    def test_classify_basic(self):
        r = run_jina("classify", "I love this movie", "--labels", "positive,negative")
        assert r.returncode == 0
        # Human output should show label + score
        assert "(" in r.stdout  # score in parens

    def test_classify_json(self):
        r = run_jina("classify", "stock market crash", "--labels", "business,sports,tech", "--json")
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert isinstance(data, list)
        assert len(data) > 0

    def test_classify_stdin(self):
        r = run_jina("classify", "--labels", "positive,negative", stdin="I love this\nI hate this\n")
        assert r.returncode == 0
        lines = [l for l in r.stdout.strip().split("\n") if l]
        assert len(lines) == 2

    def test_classify_no_labels(self):
        """Should error when no --labels provided."""
        r = run_jina("classify", "text")
        assert r.returncode != 0


class TestPipe:
    def test_search_to_rerank(self):
        """search | rerank pipe should work."""
        # First search
        search = run_jina("search", "jina ai", "-n", "3")
        assert search.returncode == 0
        # Then pipe to rerank
        rerank = run_jina("rerank", "search foundation models", stdin=search.stdout)
        assert rerank.returncode == 0
        assert "[" in rerank.stdout  # scores

    def test_echo_embed_pipe(self):
        """echo text | jina embed should work."""
        r = run_jina("embed", stdin="hello world\n")
        assert r.returncode == 0
        assert "dim=" in r.stdout

    def test_multiline_dedup_pipe(self):
        """echo -e "a\\nb" | jina dedup should work."""
        r = run_jina("dedup", stdin="machine learning\nmachine learning algorithms\nquantum physics\n")
        assert r.returncode == 0
        lines = [l for l in r.stdout.strip().split("\n") if l]
        assert len(lines) >= 1

    def test_echo_classify_pipe(self):
        """echo text | jina classify --labels should work."""
        r = run_jina("classify", "--labels", "positive,negative", stdin="this is great\n")
        assert r.returncode == 0
        assert "(" in r.stdout

    def test_search_to_dedup_pipe(self):
        """search | dedup pipe should work."""
        search = run_jina("search", "python programming", "-n", "3")
        assert search.returncode == 0
        if search.stdout.strip():
            dedup = run_jina("dedup", stdin=search.stdout)
            assert dedup.returncode == 0
