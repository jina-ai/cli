"""Shared HTTP client for Jina AI APIs."""

import os
import sys
import json
import time

import httpx

from jina_cli import __version__

API_BASE = "https://api.jina.ai"
READER_BASE = "https://r.jina.ai"
SEARCH_SVIP_BASE = "https://svip.jina.ai"
DEFAULT_TIMEOUT = 30.0

USER_AGENT = f"jina-cli/{__version__}"

# Retry config for transient errors (429, 5xx, timeouts, connection errors)
MAX_RETRIES = 3
RETRY_BACKOFF = [1.0, 2.0, 4.0]  # seconds between retries

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def get_api_key(api_key: str | None = None) -> str | None:
    """Get API key from argument, env var, or return None."""
    return api_key or os.environ.get("JINA_API_KEY")


def require_api_key(api_key: str | None = None) -> str:
    """Get API key or exit with helpful error."""
    key = get_api_key(api_key)
    if not key:
        print(
            "Error: API key required.\n"
            "Fix: export JINA_API_KEY=your-key\n"
            "     Or pass --api-key your-key\n"
            "Get a free key: https://jina.ai/?sui=apikey",
            file=sys.stderr,
        )
        sys.exit(1)
    return key


def _client(timeout: float = DEFAULT_TIMEOUT) -> httpx.Client:
    return httpx.Client(timeout=timeout, follow_redirects=True)


def _auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def _base_headers() -> dict[str, str]:
    """Headers included in every request."""
    return {"User-Agent": USER_AGENT}


def _request_with_retry(
    method: str,
    url: str,
    client: httpx.Client,
    **kwargs,
) -> httpx.Response:
    """Execute an HTTP request with retry on transient failures.

    Retries on: 429, 5xx status codes, timeouts, and connection errors.
    Uses exponential backoff between retries.
    """
    headers = kwargs.pop("headers", {})
    headers = {**_base_headers(), **headers}
    kwargs["headers"] = headers

    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            if method == "GET":
                resp = client.get(url, **kwargs)
            else:
                resp = client.post(url, **kwargs)

            if resp.status_code not in _RETRYABLE_STATUS or attempt == MAX_RETRIES - 1:
                resp.raise_for_status()
                return resp

            # Retryable status - wait and retry
            last_exc = httpx.HTTPStatusError(
                f"HTTP {resp.status_code}", request=resp.request, response=resp
            )
            wait = RETRY_BACKOFF[attempt]
            if resp.status_code == 429:
                # Respect Retry-After header if present
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait = max(wait, float(retry_after))
                    except ValueError:
                        pass
            print(f"Retrying ({attempt + 1}/{MAX_RETRIES}) after {wait:.0f}s...", file=sys.stderr)
            time.sleep(wait)

        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            last_exc = exc
            if attempt == MAX_RETRIES - 1:
                raise
            wait = RETRY_BACKOFF[attempt]
            print(f"Retrying ({attempt + 1}/{MAX_RETRIES}) after {wait:.0f}s...", file=sys.stderr)
            time.sleep(wait)

    raise last_exc


# -- Reader API --


def read_url(
    url: str,
    api_key: str | None = None,
    with_links: bool = False,
    with_images: bool = False,
    as_json: bool = False,
) -> dict | str:
    """Read a URL and extract clean content via r.jina.ai."""
    headers = {
        "Accept": "application/json" if as_json else "text/markdown",
        "Content-Type": "application/json",
        "X-Md-Link-Style": "discarded",
    }
    if with_links:
        headers["X-With-Links-Summary"] = "all"
    if with_images:
        headers["X-With-Images-Summary"] = "true"
    else:
        headers["X-Retain-Images"] = "none"

    key = get_api_key(api_key)
    if key:
        headers.update(_auth_headers(key))

    with _client() as client:
        resp = _request_with_retry(
            "POST", f"{READER_BASE}/",
            client, headers=headers, json={"url": url},
        )

    if as_json:
        return resp.json()
    return resp.text


# -- Screenshot API --


def screenshot_url(
    url: str,
    api_key: str | None = None,
    full_page: bool = False,
) -> dict:
    """Capture screenshot of a URL via r.jina.ai."""
    key = require_api_key(api_key)
    fmt = "pageshot" if full_page else "screenshot"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "X-Return-Format": fmt,
        **_auth_headers(key),
    }

    with _client(timeout=60.0) as client:
        resp = _request_with_retry(
            "POST", f"{READER_BASE}/",
            client, headers=headers, json={"url": url},
        )
    return resp.json()


# -- Search API --


def search_web(
    query: str,
    api_key: str | None = None,
    num: int = 5,
    tbs: str | None = None,
    location: str | None = None,
    gl: str | None = None,
    hl: str | None = None,
    as_json: bool = False,
) -> dict | str:
    """Web search via s.jina.ai."""
    key = require_api_key(api_key)
    headers = {
        "Accept": "application/json" if as_json else "text/markdown",
        "Content-Type": "application/json",
        **_auth_headers(key),
    }

    body: dict = {"q": query, "num": num}
    if tbs:
        body["tbs"] = tbs
    if location:
        body["location"] = location
    if gl:
        body["gl"] = gl
    if hl:
        body["hl"] = hl

    with _client() as client:
        resp = _request_with_retry(
            "POST", f"{SEARCH_SVIP_BASE}/",
            client, headers=headers, json=body,
        )

    if as_json:
        return resp.json()
    return resp.text


def search_arxiv(
    query: str,
    api_key: str | None = None,
    num: int = 5,
    tbs: str | None = None,
    as_json: bool = False,
) -> dict | str:
    """Search arXiv papers."""
    key = require_api_key(api_key)
    headers = {
        "Accept": "application/json" if as_json else "text/markdown",
        "Content-Type": "application/json",
        **_auth_headers(key),
    }

    body: dict = {"q": query, "domain": "arxiv", "num": num}
    if tbs:
        body["tbs"] = tbs

    with _client() as client:
        resp = _request_with_retry(
            "POST", f"{SEARCH_SVIP_BASE}/",
            client, headers=headers, json=body,
        )

    if as_json:
        return resp.json()
    return resp.text


def search_ssrn(
    query: str,
    api_key: str | None = None,
    num: int = 5,
    tbs: str | None = None,
    as_json: bool = False,
) -> dict | str:
    """Search SSRN papers."""
    key = require_api_key(api_key)
    headers = {
        "Accept": "application/json" if as_json else "text/markdown",
        "Content-Type": "application/json",
        **_auth_headers(key),
    }

    body: dict = {"q": query, "domain": "ssrn", "num": num}
    if tbs:
        body["tbs"] = tbs

    with _client() as client:
        resp = _request_with_retry(
            "POST", f"{SEARCH_SVIP_BASE}/",
            client, headers=headers, json=body,
        )

    if as_json:
        return resp.json()
    return resp.text


def search_images(
    query: str,
    api_key: str | None = None,
    num: int = 5,
    tbs: str | None = None,
    gl: str | None = None,
    hl: str | None = None,
    as_json: bool = False,
) -> dict | str:
    """Search images."""
    key = require_api_key(api_key)
    headers = {
        "Accept": "application/json" if as_json else "text/markdown",
        "Content-Type": "application/json",
        **_auth_headers(key),
    }

    body: dict = {"q": query, "type": "images", "num": num}
    if tbs:
        body["tbs"] = tbs
    if gl:
        body["gl"] = gl
    if hl:
        body["hl"] = hl

    with _client() as client:
        resp = _request_with_retry(
            "POST", f"{SEARCH_SVIP_BASE}/",
            client, headers=headers, json=body,
        )

    if as_json:
        return resp.json()
    return resp.text


def search_blog(
    query: str,
    api_key: str | None = None,
    num: int = 5,
    tbs: str | None = None,
    as_json: bool = False,
) -> dict | str:
    """Search Jina AI blog via web search with site: filter."""
    key = require_api_key(api_key)
    site_query = f"site:jina.ai/news {query}"
    return search_web(site_query, api_key=key, num=num, tbs=tbs, as_json=as_json)


# -- Expand Query --


def expand_query(
    query: str,
    api_key: str | None = None,
) -> list[str]:
    """Expand a search query into related queries."""
    key = require_api_key(api_key)
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        **_auth_headers(key),
    }

    with _client() as client:
        resp = _request_with_retry(
            "POST", f"{SEARCH_SVIP_BASE}/",
            client, headers=headers,
            json={"q": query, "query_expansion": True},
        )
    data = resp.json()
    return data.get("results", data.get("data", []))


# -- Embeddings API --


def embed(
    texts: list[str],
    api_key: str | None = None,
    model: str = "jina-embeddings-v3",
    task: str = "text-matching",
    dimensions: int | None = None,
    late_chunking: bool = False,
) -> list[dict]:
    """Generate embeddings for texts."""
    key = require_api_key(api_key)
    headers = {
        "Content-Type": "application/json",
        **_auth_headers(key),
    }

    body: dict = {
        "model": model,
        "task": task,
        "input": texts,
    }
    if dimensions:
        body["dimensions"] = dimensions
    if late_chunking:
        body["late_chunking"] = True

    with _client() as client:
        resp = _request_with_retry(
            "POST", f"{API_BASE}/v1/embeddings",
            client, headers=headers, json=body,
        )
    data = resp.json()
    return data.get("data", [])


# -- Local Embeddings (via jina-grep server) --


LOCAL_SERVER = "http://localhost:8089"


def local_embed(
    texts: list[str],
    model: str = "jina-embeddings-v5-nano",
    task: str = "text-matching",
) -> list[dict]:
    """Generate embeddings via local jina-grep server on localhost:8089."""
    body = {
        "input": texts,
        "model": model,
        "task": task,
    }
    try:
        with _client(timeout=120.0) as client:
            resp = _request_with_retry(
                "POST", f"{LOCAL_SERVER}/v1/embeddings",
                client, json=body,
            )
    except httpx.ConnectError:
        print(
            "Error: local embedding server not running.\n"
            "Fix: jina-grep serve start",
            file=sys.stderr,
        )
        sys.exit(1)
    data = resp.json()
    return data.get("data", [])


def local_rerank(
    query: str,
    documents: list[str],
    model: str = "jina-embeddings-v5-nano",
    task: str = "text-matching",
    top_n: int | None = None,
) -> list[dict]:
    """Rerank documents using local embeddings and cosine similarity."""
    all_texts = [query] + documents
    embeddings_data = local_embed(all_texts, model=model, task=task)
    embeddings = [item["embedding"] for item in embeddings_data]

    query_emb = embeddings[0]
    doc_embs = embeddings[1:]

    scored = []
    for i, doc_emb in enumerate(doc_embs):
        score = _cosine_similarity(query_emb, doc_emb)
        scored.append({"index": i, "relevance_score": score})

    scored.sort(key=lambda x: -x["relevance_score"])

    if top_n is not None:
        scored = scored[:top_n]

    return scored


# -- Reranker API --


def rerank(
    query: str,
    documents: list[str],
    api_key: str | None = None,
    model: str = "jina-reranker-v3",
    top_n: int | None = None,
) -> list[dict]:
    """Rerank documents by relevance to query."""
    key = require_api_key(api_key)
    headers = {
        "Content-Type": "application/json",
        **_auth_headers(key),
    }

    body: dict = {
        "model": model,
        "query": query,
        "documents": documents,
    }
    if top_n is not None:
        body["top_n"] = top_n

    with _client() as client:
        resp = _request_with_retry(
            "POST", f"{API_BASE}/v1/rerank",
            client, headers=headers, json=body,
        )
    data = resp.json()
    return data.get("results", [])


# -- Deduplication --


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def deduplicate(
    strings: list[str],
    api_key: str | None = None,
    k: int | None = None,
) -> list[dict]:
    """Deduplicate strings using embeddings + greedy selection.

    Uses facility-location submodular optimization:
    greedily selects items that maximize coverage diversity.
    """
    if not strings:
        return []
    if len(strings) == 1:
        return [{"index": 0, "text": strings[0]}]

    key = require_api_key(api_key)

    # Get embeddings (v5-text-small is faster and sufficient for dedup)
    embeddings_data = embed(
        strings,
        api_key=key,
        model="jina-embeddings-v5-text-small",
        task="text-matching",
    )
    embeddings = [item["embedding"] for item in embeddings_data]

    n = len(embeddings)

    # Compute similarity matrix
    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            s = _cosine_similarity(embeddings[i], embeddings[j])
            sim[i][j] = s
            sim[j][i] = s

    # Lazy greedy submodular selection
    if k is None:
        # Auto-detect: keep adding until marginal gain drops below threshold
        threshold = 1e-2
        k = n

    selected: list[int] = []
    # Track max similarity of each item to any selected item (facility-location)
    coverage = [0.0] * n

    for _ in range(min(k, n)):
        best_idx = -1
        best_gain = -1.0

        for i in range(n):
            if i in selected:
                continue
            # Marginal gain: how much does adding i improve coverage?
            gain = 0.0
            for j in range(n):
                new_cov = max(coverage[j], sim[j][i])
                gain += new_cov - coverage[j]

            if gain > best_gain:
                best_gain = gain
                best_idx = i

        if best_idx == -1 or (len(selected) > 0 and best_gain < threshold):
            break

        selected.append(best_idx)
        for j in range(n):
            coverage[j] = max(coverage[j], sim[j][best_idx])

    return [{"index": i, "text": strings[i]} for i in selected]


# -- BibTeX Search --


def search_bibtex(
    query: str,
    api_key: str | None = None,
    author: str | None = None,
    year: int | None = None,
    num: int = 10,
) -> list[dict]:
    """Search for BibTeX entries via DBLP and Semantic Scholar."""

    def _search_dblp() -> list[dict]:
        q = f"{query} {author}" if author else query
        params = {"q": q, "format": "json", "h": str(min(num * 2, 100))}
        try:
            with _client(timeout=5.0) as client:
                resp = _request_with_retry(
                    "GET", "https://dblp.org/search/publ/api",
                    client, params=params,
                )
            data = resp.json()
            hits = data.get("result", {}).get("hits", {}).get("hit", [])
            results = []
            for hit in hits:
                info = hit.get("info", {})
                authors_data = info.get("authors", {}).get("author", [])
                if isinstance(authors_data, dict):
                    authors_data = [authors_data]
                authors = [a.get("text", a) if isinstance(a, dict) else str(a) for a in authors_data]
                results.append(
                    {
                        "title": info.get("title", ""),
                        "authors": authors,
                        "year": int(info.get("year", 0)),
                        "venue": info.get("venue", ""),
                        "doi": info.get("doi", ""),
                        "url": info.get("ee", info.get("url", "")),
                        "type": info.get("type", ""),
                    }
                )
            return results
        except Exception:
            return []

    def _search_semantic_scholar() -> list[dict]:
        params: dict = {
            "query": query,
            "limit": str(min(num * 2, 100)),
            "fields": "title,authors,year,venue,externalIds,abstract,citationCount,url",
        }
        if year:
            params["year"] = f"{year}-"
        try:
            with _client(timeout=5.0) as client:
                resp = _request_with_retry(
                    "GET", "https://api.semanticscholar.org/graph/v1/paper/search",
                    client, params=params,
                )
            data = resp.json()
            papers = data.get("data", [])
            results = []
            for p in papers:
                ext = p.get("externalIds", {}) or {}
                authors = [a.get("name", "") for a in (p.get("authors") or [])]
                results.append(
                    {
                        "title": p.get("title", ""),
                        "authors": authors,
                        "year": p.get("year", 0) or 0,
                        "venue": p.get("venue", ""),
                        "doi": ext.get("DOI", ""),
                        "arxiv_id": ext.get("ArXiv", ""),
                        "citations": p.get("citationCount", 0),
                        "abstract": p.get("abstract", ""),
                        "url": p.get("url", ""),
                    }
                )
            return results
        except Exception:
            return []

    # Run both in sequence (could be threaded but keep simple)
    dblp = _search_dblp()
    ss = _search_semantic_scholar()

    # Merge and deduplicate
    seen_titles: dict[str, dict] = {}
    all_results = dblp + ss

    for r in all_results:
        title_key = r.get("title", "").lower().strip()
        if not title_key:
            continue
        if title_key in seen_titles:
            existing = seen_titles[title_key]
            # Merge: prefer longer abstract, higher citation count
            if len(r.get("abstract", "")) > len(existing.get("abstract", "")):
                existing["abstract"] = r["abstract"]
            if r.get("citations", 0) > existing.get("citations", 0):
                existing["citations"] = r["citations"]
            if r.get("doi") and not existing.get("doi"):
                existing["doi"] = r["doi"]
            if r.get("arxiv_id") and not existing.get("arxiv_id"):
                existing["arxiv_id"] = r["arxiv_id"]
        else:
            seen_titles[title_key] = r

    results = list(seen_titles.values())[:num]

    # Generate BibTeX
    for r in results:
        r["bibtex"] = _make_bibtex(r)

    return results


def _escape_bibtex(s: str) -> str:
    for ch in ("&", "%", "_", "$", "#"):
        s = s.replace(ch, f"\\{ch}")
    return s


def _make_bibtex(entry: dict) -> str:
    authors = entry.get("authors", [])
    year = entry.get("year", 0)
    title = entry.get("title", "")

    # Generate key
    first_author = authors[0].split()[-1].lower() if authors else "unknown"
    first_word = "".join(c for c in title.split()[0].lower() if c.isalpha()) if title else "untitled"
    key = f"{first_author}{year}{first_word}"

    # Determine type
    venue = entry.get("venue", "").lower()
    if entry.get("arxiv_id"):
        entry_type = "article"
    elif any(kw in venue for kw in ("proc", "conf", "workshop", "sympos")):
        entry_type = "inproceedings"
    elif "journal" in venue or "trans" in venue:
        entry_type = "article"
    else:
        entry_type = "misc"

    lines = [f"@{entry_type}{{{key},"]
    lines.append(f"  title = {{{_escape_bibtex(title)}}},")

    if authors:
        author_str = " and ".join(authors)
        lines.append(f"  author = {{{_escape_bibtex(author_str)}}},")
    if year:
        lines.append(f"  year = {{{year}}},")
    if entry.get("venue"):
        field = "booktitle" if entry_type == "inproceedings" else "journal"
        lines.append(f"  {field} = {{{_escape_bibtex(entry['venue'])}}},")
    if entry.get("doi"):
        lines.append(f"  doi = {{{entry['doi']}}},")
    if entry.get("url"):
        lines.append(f"  url = {{{entry['url']}}},")
    if entry.get("arxiv_id"):
        lines.append(f"  eprint = {{{entry['arxiv_id']}}},")
        lines.append("  archivePrefix = {arXiv},")

    lines.append("}")
    return "\n".join(lines)


# -- PDF Extraction --


def extract_pdf(
    url: str | None = None,
    arxiv_id: str | None = None,
    api_key: str | None = None,
    max_edge: int = 1024,
    extract_type: str | None = None,
) -> dict:
    """Extract figures, tables, equations from a PDF."""
    key = require_api_key(api_key)
    headers = {
        "Content-Type": "application/json",
        **_auth_headers(key),
    }

    body: dict = {"max_edge": max_edge}
    if arxiv_id:
        body["id"] = arxiv_id
    elif url:
        body["url"] = url
    else:
        print("Error: provide either --url or --arxiv-id", file=sys.stderr)
        sys.exit(1)

    if extract_type:
        body["type"] = extract_type

    with _client(timeout=60.0) as client:
        resp = _request_with_retry(
            "POST", f"{SEARCH_SVIP_BASE}/extract-pdf",
            client, headers=headers, json=body,
        )
    return resp.json()


# -- Datetime Guess --


def guess_datetime(url: str) -> dict:
    """Guess the publish/update datetime of a URL.

    This calls the Jina reader API which handles the detection server-side.
    """
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "X-Return-Format": "datetime",
    }

    key = get_api_key()
    if key:
        headers.update(_auth_headers(key))

    with _client() as client:
        resp = _request_with_retry(
            "POST", f"{READER_BASE}/",
            client, headers=headers, json={"url": url},
        )
    return resp.json()


# -- Primer --


def primer() -> dict:
    """Get context info (time, location, network)."""
    headers = {"Accept": "application/json"}
    key = get_api_key()
    if key:
        headers.update(_auth_headers(key))

    with _client() as client:
        resp = _request_with_retry(
            "GET", f"{READER_BASE}/",
            client, headers=headers,
        )
    return resp.json()
