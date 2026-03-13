"""Helpers for output formatting and stdin reading."""

import sys
import json
import signal

# -- Exit codes --
# Meaningful exit codes for scripting and agent use.
# 0 = success, 1 = user/input error, 2 = API/server error, 130 = interrupted (Ctrl+C)
EXIT_OK = 0
EXIT_USER_ERROR = 1
EXIT_API_ERROR = 2
EXIT_INTERRUPTED = 130


def setup_signals():
    """Install signal handlers for clean Ctrl+C (no stack traces)."""
    def _handler(sig, frame):
        # Write newline so prompt isn't mangled, then exit quietly
        print("", file=sys.stderr)
        sys.exit(EXIT_INTERRUPTED)
    signal.signal(signal.SIGINT, _handler)
    # Ignore SIGPIPE so piping to head/tail doesn't traceback
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)


def read_stdin_lines() -> list[str]:
    """Read lines from stdin if available (pipe mode)."""
    if sys.stdin.isatty():
        return []
    lines = []
    for line in sys.stdin:
        stripped = line.rstrip("\n")
        if stripped:
            lines.append(stripped)
    return lines


def format_bibtex_results(results: list[dict], as_json: bool = False) -> str:
    """Format BibTeX results for display."""
    if as_json:
        return json.dumps(results, indent=2, ensure_ascii=False)

    lines = []
    for r in results:
        bibtex = r.get("bibtex", "")
        if bibtex:
            lines.append(bibtex)
            lines.append("")
    return "\n".join(lines)


def format_rerank_results(
    results: list[dict],
    documents: list[str],
    as_json: bool = False,
) -> str:
    """Format rerank results for display."""
    if as_json:
        return json.dumps(results, indent=2, ensure_ascii=False)

    lines = []
    for r in results:
        idx = r.get("index", 0)
        score = r.get("relevance_score", r.get("score", 0))
        text = r.get("document", {}).get("text", "") if isinstance(r.get("document"), dict) else ""
        if not text and idx < len(documents):
            text = documents[idx]
        # Truncate long lines
        if len(text) > 200:
            text = text[:200] + "..."
        lines.append(f"[{score:.4f}] {text}")
    return "\n".join(lines)


def format_embeddings(data: list[dict], as_json: bool = False) -> str:
    """Format embedding results for display."""
    if as_json:
        return json.dumps(data, indent=2, ensure_ascii=False)

    lines = []
    for item in data:
        idx = item.get("index", 0)
        embedding = item.get("embedding", [])
        dim = len(embedding)
        # Show first few values
        preview = embedding[:5]
        preview_str = ", ".join(f"{v:.6f}" for v in preview)
        lines.append(f"[{idx}] dim={dim} [{preview_str}, ...]")
    return "\n".join(lines)


def format_dedup_results(results: list[dict], as_json: bool = False) -> str:
    """Format deduplication results for display."""
    if as_json:
        return json.dumps(results, indent=2, ensure_ascii=False)

    lines = []
    for r in results:
        lines.append(r.get("text", ""))
    return "\n".join(lines)


def format_pdf_results(data: dict, as_json: bool = False) -> str:
    """Format PDF extraction results for display."""
    if as_json:
        # Strip base64 image data for json output to keep it manageable
        clean = dict(data)
        floats = []
        for f in clean.get("floats", []):
            cf = dict(f)
            if "image" in cf:
                cf["image"] = f"<base64 {len(cf['image'])} chars>"
            floats.append(cf)
        clean["floats"] = floats
        return json.dumps(clean, indent=2, ensure_ascii=False)

    meta = data.get("meta", {})
    floats = data.get("floats", [])

    lines = []
    lines.append(f"Pages: {meta.get('num_pages', '?')}")
    lines.append(f"Extracted items: {meta.get('num_floats', len(floats))}")
    lines.append("")

    for f in floats:
        ftype = f.get("type", "unknown")
        number = f.get("number", "")
        caption = f.get("caption", "")
        page = f.get("page", "?")
        lines.append(f"  [{ftype} {number}] page {page}")
        if caption:
            lines.append(f"    {caption}")
    return "\n".join(lines)


def handle_http_error(e: Exception) -> None:
    """Handle HTTP errors with actionable guidance.

    Exit codes:
      1 = user/input error (bad args, missing key, invalid request)
      2 = API/server error (server down, rate limit, timeout, network)
    """
    import httpx

    if isinstance(e, httpx.HTTPStatusError):
        status = e.response.status_code
        try:
            body = e.response.text[:500]
        except Exception:
            body = ""

        if status == 401:
            print(
                "Error: invalid or expired API key.\n"
                "Fix: export JINA_API_KEY=your-key\n"
                "     Or pass --api-key your-key\n"
                "Get a free key: https://jina.ai/?sui=apikey",
                file=sys.stderr,
            )
            sys.exit(EXIT_USER_ERROR)
        elif status == 402:
            print(
                "Error: API quota exhausted.\n"
                "Fix: top up credits at https://jina.ai/api-dashboard/billing\n"
                "     Or check usage at https://jina.ai/api-dashboard",
                file=sys.stderr,
            )
            sys.exit(EXIT_USER_ERROR)
        elif status == 422:
            print(
                f"Error: invalid request parameters.\n"
                f"Server said: {body}\n"
                f"Fix: check your arguments with --help",
                file=sys.stderr,
            )
            sys.exit(EXIT_USER_ERROR)
        elif status == 429:
            print(
                "Error: rate limit hit.\n"
                "Fix: wait a few seconds and retry\n"
                "     Or add an API key for higher limits: export JINA_API_KEY=your-key\n"
                "     Get a key: https://jina.ai/?sui=apikey",
                file=sys.stderr,
            )
            sys.exit(EXIT_API_ERROR)
        elif status >= 500:
            print(
                f"Error: Jina API server error (HTTP {status}).\n"
                f"Server said: {body}\n"
                f"Fix: retry in a moment. If persistent, check https://status.jina.ai",
                file=sys.stderr,
            )
            sys.exit(EXIT_API_ERROR)
        else:
            print(
                f"Error: HTTP {status}.\n"
                f"Server said: {body}\n"
                f"Fix: check your arguments with --help",
                file=sys.stderr,
            )
            sys.exit(EXIT_API_ERROR)
    elif isinstance(e, httpx.ConnectError):
        print(
            "Error: cannot connect to Jina API.\n"
            "Fix: check your internet connection\n"
            "     Jina API status: https://status.jina.ai",
            file=sys.stderr,
        )
        sys.exit(EXIT_API_ERROR)
    elif isinstance(e, httpx.TimeoutException):
        print(
            "Error: request timed out.\n"
            "Fix: retry the command. For large inputs, try smaller batches",
            file=sys.stderr,
        )
        sys.exit(EXIT_API_ERROR)
    else:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(EXIT_API_ERROR)
