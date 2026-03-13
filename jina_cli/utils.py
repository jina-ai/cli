"""Helpers for output formatting and stdin reading."""

import sys
import json
import select


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


def has_stdin() -> bool:
    """Check if there is data available on stdin."""
    if sys.stdin.isatty():
        return False
    ready, _, _ = select.select([sys.stdin], [], [], 0.0)
    return bool(ready)


def format_search_results(results: list[dict], as_json: bool = False) -> str:
    """Format search results for display."""
    if as_json:
        return json.dumps(results, indent=2, ensure_ascii=False)

    lines = []
    for i, r in enumerate(results):
        title = r.get("title", "")
        url = r.get("url", "")
        snippet = r.get("snippet", r.get("description", ""))
        date = r.get("date", "")

        lines.append(f"[{i + 1}] {title}")
        if url:
            lines.append(f"    {url}")
        if date:
            lines.append(f"    {date}")
        if snippet:
            lines.append(f"    {snippet}")
        lines.append("")
    return "\n".join(lines)


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
    """Handle HTTP errors with helpful messages."""
    import httpx

    if isinstance(e, httpx.HTTPStatusError):
        status = e.response.status_code
        if status == 401:
            print(
                "Error: Authentication failed. Check your API key.\n"
                "Get a key at https://jina.ai/?sui=apikey",
                file=sys.stderr,
            )
        elif status == 402:
            print(
                "Error: Out of quota. Top up at https://jina.ai",
                file=sys.stderr,
            )
        elif status == 429:
            print(
                "Error: Rate limit exceeded. Try again later or upgrade at https://jina.ai",
                file=sys.stderr,
            )
        else:
            try:
                body = e.response.text
            except Exception:
                body = ""
            print(f"Error: HTTP {status}: {body}", file=sys.stderr)
    else:
        print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
