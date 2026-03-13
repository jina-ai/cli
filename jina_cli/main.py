"""Jina CLI - all Jina AI APIs in one command line tool.

Usage:
    jina read URL              Read and extract content from a URL
    jina search QUERY          Web search
    jina embed TEXT             Generate embeddings
    jina rerank QUERY          Rerank documents by relevance
    jina dedup                 Deduplicate text lines from stdin
    jina screenshot URL        Capture screenshot of a URL
    jina bibtex QUERY          Search for BibTeX entries
    jina expand QUERY          Expand a search query
    jina pdf URL               Extract figures/tables from PDF
    jina datetime URL          Guess publish date of a URL
    jina primer                Get context info
    jina grep PATTERN          Semantic grep (requires: pip install jina-grep)

Exit codes:
    0  success
    1  user/input error (missing args, bad input, missing API key)
    2  API/server error (network, timeout, server error)
    130  interrupted (Ctrl+C)
"""

import sys
import json

import click

from jina_cli import __version__
from jina_cli import api, utils
from jina_cli.utils import EXIT_OK, EXIT_USER_ERROR, EXIT_API_ERROR


class AliasedGroup(click.Group):
    """Click group that suggests similar commands on typos."""

    def resolve_command(self, ctx, args):
        try:
            return super().resolve_command(ctx, args)
        except click.UsageError:
            cmd_name = args[0] if args else ""
            matches = []
            for name in self.list_commands(ctx):
                if name.startswith(cmd_name[:2]):
                    matches.append(name)
            if matches:
                suggestion = ", ".join(matches)
                raise click.UsageError(
                    f"Unknown command '{cmd_name}'. Did you mean: {suggestion}?\n"
                    f"Run 'jina --help' to see all commands."
                )
            raise click.UsageError(
                f"Unknown command '{cmd_name}'. Run 'jina --help' to see all commands."
            )


def _short_usage(usage: str, examples: list[str]) -> None:
    """Print short usage (layer 1) instead of full --help (layer 2).

    Progressive disclosure:
      Layer 0: `jina`           -> command list
      Layer 1: `jina search`    -> usage + examples (this function)
      Layer 2: `jina search -h` -> full options
    """
    lines = [usage, ""]
    for ex in examples:
        lines.append(f"  {ex}")
    lines.append("")
    lines.append("Run with --help for all options.")
    click.echo("\n".join(lines), err=True)
    sys.exit(EXIT_USER_ERROR)


def _validate_url(url: str) -> str:
    """Validate that input looks like a URL. Returns the URL or exits with guidance."""
    if not url.startswith(("http://", "https://")):
        click.echo(
            f"Error: '{url}' is not a valid URL.\n"
            f"Fix: URLs must start with http:// or https://",
            err=True,
        )
        sys.exit(EXIT_USER_ERROR)
    return url


@click.group(cls=AliasedGroup, invoke_without_command=True)
@click.version_option(__version__, prog_name="jina")
@click.option("--api-key", envvar="JINA_API_KEY", default=None, hidden=True,
              help="Jina API key (or set JINA_API_KEY env var)")
@click.pass_context
def cli(ctx, api_key):
    """Jina AI CLI - search, read, embed, rerank, and more.

    All Jina AI APIs as Unix-friendly commands. Supports pipes and chaining.

    \b
    Environment variables:
        JINA_API_KEY    API key for Jina services (required for most commands)

    Set your API key: export JINA_API_KEY=your-key
    Get a key: https://jina.ai/?sui=apikey
    """
    utils.setup_signals()
    ctx.ensure_object(dict)
    ctx.obj["api_key"] = api_key
    if ctx.invoked_subcommand is None:
        click.echo(
            "jina - all Jina AI APIs in one command\n"
            "\n"
            "  jina read URL              Extract markdown from web pages\n"
            "  jina search QUERY          Web search (also --arxiv, --ssrn, --images, --blog)\n"
            "  jina embed TEXT             Generate embeddings\n"
            "  jina rerank QUERY          Rerank documents from stdin by relevance\n"
            "  jina dedup                 Deduplicate text from stdin\n"
            "  jina screenshot URL        Capture screenshot of a URL\n"
            "  jina bibtex QUERY          Search BibTeX citations\n"
            "  jina expand QUERY          Expand query into related queries\n"
            "  jina pdf URL               Extract figures/tables from PDFs\n"
            "  jina datetime URL          Guess publish/update date of a URL\n"
            "  jina primer                Context info (time, location, network)\n"
            "  jina grep PATTERN          Semantic grep (requires: pip install jina-grep)\n"
            "\n"
            "Run any command without arguments for usage examples.\n"
            "Run any command with --help for full options.\n"
            "API key: export JINA_API_KEY=your-key (https://jina.ai/?sui=apikey)",
            err=True,
        )


# -- read --


@cli.command()
@click.argument("url", required=False)
@click.option("--links", is_flag=True, help="Include hyperlinks in output")
@click.option("--images", is_flag=True, help="Include images in output")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--api-key", default=None, help="Jina API key")
@click.pass_context
def read(ctx, url, links, images, as_json, api_key):
    """Read a URL and extract clean content.

    Extracts readable text/markdown from any web page.

    \b
    Examples:
        jina read https://example.com
        echo "https://example.com" | jina read
        jina read https://example.com --json
        cat urls.txt | jina read
    """
    key = api_key or ctx.obj.get("api_key")

    # Read URL from stdin if not provided
    urls = []
    if url:
        urls.append(url)
    else:
        stdin_lines = utils.read_stdin_lines()
        if stdin_lines:
            urls = [line.strip() for line in stdin_lines if line.strip().startswith(("http://", "https://"))]

    if not urls:
        _short_usage(
            "Usage: jina read URL",
            ["jina read https://example.com",
             "echo URL | jina read",
             "cat urls.txt | jina read"],
        )

    for u in urls:
        _validate_url(u)

    try:
        for i, u in enumerate(urls):
            result = api.read_url(u, api_key=key, with_links=links, with_images=images, as_json=as_json)
            if as_json:
                click.echo(json.dumps(result, indent=2, ensure_ascii=False) if isinstance(result, dict) else result)
            else:
                click.echo(result)
    except Exception as e:
        utils.handle_http_error(e)


# -- search --


@cli.command()
@click.argument("query", required=False)
@click.option("--arxiv", is_flag=True, help="Search arXiv papers")
@click.option("--ssrn", is_flag=True, help="Search SSRN papers")
@click.option("--images", is_flag=True, help="Search images")
@click.option("--blog", is_flag=True, help="Search Jina AI blog")
@click.option("-n", "--num", default=5, type=int, help="Number of results (default: 5)")
@click.option("--time", "tbs", type=click.Choice(["h", "d", "w", "m", "y"]),
              help="Time filter: h(our), d(ay), w(eek), m(onth), y(ear)")
@click.option("--location", default=None, help="Location for search results")
@click.option("--gl", default=None, help="Country code (e.g. us, de, jp)")
@click.option("--hl", default=None, help="Language code (e.g. en, zh-cn)")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--api-key", default=None, help="Jina API key")
@click.pass_context
def search(ctx, query, arxiv, ssrn, images, blog, num, tbs, location, gl, hl, as_json, api_key):
    """Search the web, arXiv, SSRN, images, or Jina blog.

    \b
    Examples:
        jina search "transformer architecture"
        jina search --arxiv "attention mechanism"
        jina search --ssrn "corporate governance"
        jina search --images "neural network diagram"
        jina search --blog "embeddings"
        jina search "AI news" --time d --num 10
        jina search "LLM" | jina rerank "embeddings"
    """
    if not query:
        stdin_lines = utils.read_stdin_lines()
        if stdin_lines:
            query = " ".join(stdin_lines)
    if not query:
        _short_usage(
            "Usage: jina search QUERY [--arxiv|--ssrn|--images|--blog]",
            ["jina search \"transformer architecture\"",
             "jina search --arxiv \"attention mechanism\"",
             "jina search \"AI\" | jina rerank \"embeddings\""],
        )

    key = api_key or ctx.obj.get("api_key")
    tbs_val = f"qdr:{tbs}" if tbs else None

    try:
        # Always fetch JSON from API, format on CLI side
        if blog:
            result = api.search_blog(query, api_key=key, num=num, tbs=tbs_val, as_json=True)
        elif arxiv:
            result = api.search_arxiv(query, api_key=key, num=num, tbs=tbs_val, as_json=True)
        elif ssrn:
            result = api.search_ssrn(query, api_key=key, num=num, tbs=tbs_val, as_json=True)
        elif images:
            result = api.search_images(query, api_key=key, num=num, tbs=tbs_val, gl=gl, hl=hl, as_json=True)
        else:
            result = api.search_web(
                query, api_key=key, num=num, tbs=tbs_val,
                location=location, gl=gl, hl=hl, as_json=True,
            )
        if as_json:
            click.echo(json.dumps(result, indent=2, ensure_ascii=False) if isinstance(result, dict) else result)
        elif isinstance(result, dict):
            for r in result.get("results", []):
                title = r.get("title", "")
                url = r.get("url", "")
                snippet = r.get("snippet", "")
                click.echo(f"{title}")
                click.echo(f"  {url}")
                if snippet:
                    click.echo(f"  {snippet}")
                click.echo()
        else:
            click.echo(result)
    except Exception as e:
        utils.handle_http_error(e)


# -- embed --


@cli.command()
@click.argument("text", nargs=-1)
@click.option("--model", default=None, help="Model name (default: jina-embeddings-v3, or v5-nano with --local)")
@click.option("--task", default=None, help="Embedding task type")
@click.option("--dimensions", type=int, default=None, help="Output dimensions (Matryoshka)")
@click.option("--local", is_flag=True, help="Use local MLX server (requires: jina-grep serve start)")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--api-key", default=None, help="Jina API key")
@click.pass_context
def embed(ctx, text, model, task, dimensions, local, as_json, api_key):
    """Generate embeddings for text.

    Input from arguments or stdin (one text per line).

    \b
    Examples:
        jina embed "hello world"
        echo "hello world" | jina embed
        jina embed "text1" "text2" "text3"
        cat texts.txt | jina embed --json
        jina embed --local "hello world"
    """
    key = api_key or ctx.obj.get("api_key")

    texts = list(text)
    if not texts:
        stdin_lines = utils.read_stdin_lines()
        texts = stdin_lines

    if not texts:
        _short_usage(
            "Usage: jina embed TEXT [TEXT ...]",
            ["jina embed \"hello world\"",
             "echo \"hello\" | jina embed",
             "jina embed --local \"hello world\"",
             "cat texts.txt | jina embed --json"],
        )

    try:
        if local:
            _model = model or "jina-embeddings-v5-nano"
            _task = task or "text-matching"
            result = api.local_embed(texts, model=_model, task=_task)
        else:
            _model = model or "jina-embeddings-v3"
            _task = task or "text-matching"
            result = api.embed(texts, api_key=key, model=_model, task=_task, dimensions=dimensions)
        click.echo(utils.format_embeddings(result, as_json=as_json))
    except Exception as e:
        utils.handle_http_error(e)


# -- rerank --


@cli.command()
@click.argument("query")
@click.option("-n", "--top-n", type=int, default=None, help="Max results to return")
@click.option("--model", default=None, help="Reranker model (default: jina-reranker-v3, or v5-nano with --local)")
@click.option("--local", is_flag=True, help="Use local MLX server (requires: jina-grep serve start)")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--api-key", default=None, help="Jina API key")
@click.pass_context
def rerank(ctx, query, top_n, model, local, as_json, api_key):
    """Rerank documents by relevance to a query.

    Reads documents from stdin, one per line.

    \b
    Examples:
        jina search "AI" | jina rerank "embeddings"
        cat docs.txt | jina rerank "machine learning"
        echo -e "doc1\\ndoc2\\ndoc3" | jina rerank "query" --top-n 2
        cat docs.txt | jina rerank --local "machine learning"
    """
    key = api_key or ctx.obj.get("api_key")
    documents = utils.read_stdin_lines()

    if not documents:
        click.echo("Error: no documents on stdin.\n"
                   "Fix: pipe text lines to rerank, one document per line.\n"
                   "  cat docs.txt | jina rerank \"your query\"\n"
                   "  jina search \"AI\" | jina rerank \"embeddings\"", err=True)
        sys.exit(EXIT_USER_ERROR)

    try:
        if local:
            _model = model or "jina-embeddings-v5-nano"
            result = api.local_rerank(query, documents, model=_model, top_n=top_n)
        else:
            _model = model or "jina-reranker-v3"
            result = api.rerank(query, documents, api_key=key, model=_model, top_n=top_n)
        click.echo(utils.format_rerank_results(result, documents, as_json=as_json))
    except Exception as e:
        utils.handle_http_error(e)


# -- dedup --


@cli.command()
@click.option("-k", type=int, default=None, help="Number of unique items to keep (auto if not set)")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--api-key", default=None, help="Jina API key")
@click.pass_context
def dedup(ctx, k, as_json, api_key):
    """Deduplicate text lines from stdin.

    Uses embeddings to find semantically unique items.

    \b
    Examples:
        cat items.txt | jina dedup
        jina search "AI" | jina dedup -k 5
    """
    key = api_key or ctx.obj.get("api_key")
    lines = utils.read_stdin_lines()

    if not lines:
        click.echo("Error: no input on stdin.\n"
                   "Fix: pipe text lines to deduplicate, one item per line.\n"
                   "  cat items.txt | jina dedup\n"
                   "  jina search \"AI\" | jina dedup -k 5", err=True)
        sys.exit(EXIT_USER_ERROR)

    try:
        result = api.deduplicate(lines, api_key=key, k=k)
        click.echo(utils.format_dedup_results(result, as_json=as_json))
    except Exception as e:
        utils.handle_http_error(e)


# -- screenshot --


@cli.command()
@click.argument("url", required=False)
@click.option("--full-page", is_flag=True, help="Capture full page (not just viewport)")
@click.option("-o", "--output", default=None, help="Save image to file")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--api-key", default=None, help="Jina API key")
@click.pass_context
def screenshot(ctx, url, full_page, output, as_json, api_key):
    """Capture a screenshot of a URL.

    \b
    Examples:
        jina screenshot https://example.com
        jina screenshot https://example.com --full-page -o page.jpg
        echo "https://example.com" | jina screenshot
    """
    key = api_key or ctx.obj.get("api_key")

    if not url:
        stdin_lines = utils.read_stdin_lines()
        if stdin_lines:
            url = stdin_lines[0].strip()

    if not url:
        _short_usage(
            "Usage: jina screenshot URL",
            ["jina screenshot https://example.com",
             "jina screenshot URL --full-page -o page.jpg"],
        )

    _validate_url(url)

    try:
        result = api.screenshot_url(url, api_key=key, full_page=full_page)
        data = result.get("data", result)

        if as_json:
            # Strip base64 image data from JSON output to keep it readable
            clean = dict(result)
            if isinstance(data, dict):
                clean_data = dict(data)
                for field in ("screenshot", "screenshotUrl", "pageshotUrl"):
                    val = clean_data.get(field, "")
                    if isinstance(val, str) and len(val) > 1000:
                        clean_data[field] = f"<base64 {len(val)} chars>"
                clean["data"] = clean_data
            click.echo(json.dumps(clean, indent=2, ensure_ascii=False))
        else:
            # Extract image URL or base64 data
            img_url = None
            img_b64 = None
            if isinstance(data, dict):
                img_url = data.get("screenshotUrl") or data.get("pageshotUrl")
                if not img_url:
                    img_b64 = data.get("screenshot", data.get("image", ""))
            if output:
                # Save to file
                import base64
                if img_url:
                    # Download from URL
                    with api._client() as client:
                        resp = client.get(img_url)
                        resp.raise_for_status()
                        with open(output, "wb") as f:
                            f.write(resp.content)
                elif img_b64:
                    if img_b64.startswith("data:"):
                        img_b64 = img_b64.split(",", 1)[1] if "," in img_b64 else img_b64
                    with open(output, "wb") as f:
                        f.write(base64.b64decode(img_b64))
                click.echo(f"Saved to {output}", err=True)
            else:
                # No output file: print URL or warn about binary
                if img_url:
                    click.echo(img_url)
                else:
                    click.echo(
                        "Error: screenshot is binary image data.\n"
                        "Fix: save to file with -o flag\n"
                        "  jina screenshot URL -o screenshot.png",
                        err=True,
                    )
                    sys.exit(EXIT_USER_ERROR)
    except Exception as e:
        utils.handle_http_error(e)


# -- bibtex --


@cli.command()
@click.argument("query", required=False)
@click.option("--author", default=None, help="Filter by author name")
@click.option("--year", type=int, default=None, help="Filter by year (minimum)")
@click.option("-n", "--num", default=10, type=int, help="Number of results (default: 10)")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--api-key", default=None, help="Jina API key")
@click.pass_context
def bibtex(ctx, query, author, year, num, as_json, api_key):
    """Search for BibTeX citations.

    Searches DBLP and Semantic Scholar, deduplicates, and outputs BibTeX entries.

    \b
    Examples:
        jina bibtex "attention is all you need"
        jina bibtex "transformer" --author Vaswani --year 2017
        jina bibtex "BERT" --json
    """
    if not query:
        stdin_lines = utils.read_stdin_lines()
        if stdin_lines:
            query = " ".join(stdin_lines)
    if not query:
        _short_usage(
            "Usage: jina bibtex QUERY",
            ["jina bibtex \"attention is all you need\"",
             "jina bibtex \"transformer\" --author Vaswani --year 2017"],
        )
    key = api_key or ctx.obj.get("api_key")

    try:
        results = api.search_bibtex(query, api_key=key, author=author, year=year, num=num)
        click.echo(utils.format_bibtex_results(results, as_json=as_json))
    except Exception as e:
        utils.handle_http_error(e)


# -- expand --


@cli.command()
@click.argument("query", required=False)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--api-key", default=None, help="Jina API key")
@click.pass_context
def expand(ctx, query, as_json, api_key):
    """Expand a search query into related queries.

    \b
    Examples:
        jina expand "machine learning"
        jina expand "climate change effects" --json
        echo "query" | jina expand
    """
    if not query:
        stdin_lines = utils.read_stdin_lines()
        if stdin_lines:
            query = " ".join(stdin_lines)
    if not query:
        _short_usage(
            "Usage: jina expand QUERY",
            ["jina expand \"machine learning\"",
             "echo \"query\" | jina expand"],
        )
    key = api_key or ctx.obj.get("api_key")

    try:
        results = api.expand_query(query, api_key=key)
        if as_json:
            click.echo(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            if isinstance(results, list):
                for r in results:
                    if isinstance(r, str):
                        click.echo(r)
                    elif isinstance(r, dict):
                        click.echo(r.get("query", r.get("text", json.dumps(r))))
                    else:
                        click.echo(str(r))
            else:
                click.echo(results)
    except Exception as e:
        utils.handle_http_error(e)


# -- pdf --


@cli.command()
@click.argument("url_or_id", required=False)
@click.option("--arxiv-id", default=None, help="arXiv paper ID (e.g. 2301.12345)")
@click.option("--type", "extract_type", default=None,
              help="Filter by type: figure, table, equation (comma-separated)")
@click.option("--max-edge", type=int, default=1024, help="Max pixel size for images (default: 1024)")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--api-key", default=None, help="Jina API key")
@click.pass_context
def pdf(ctx, url_or_id, arxiv_id, extract_type, max_edge, as_json, api_key):
    """Extract figures, tables, and equations from a PDF.

    \b
    Examples:
        jina pdf https://arxiv.org/pdf/2301.12345
        jina pdf --arxiv-id 2301.12345
        jina pdf https://example.com/paper.pdf --type figure,table
    """
    key = api_key or ctx.obj.get("api_key")

    url = None
    if url_or_id:
        # Check if it looks like an arXiv ID
        if not url_or_id.startswith("http") and not arxiv_id:
            arxiv_id = url_or_id
        else:
            url = url_or_id

    if not url and not arxiv_id:
        stdin_lines = utils.read_stdin_lines()
        if stdin_lines:
            val = stdin_lines[0].strip()
            if val.startswith("http"):
                url = val
            else:
                arxiv_id = val

    if not url and not arxiv_id:
        _short_usage(
            "Usage: jina pdf URL_OR_ARXIV_ID",
            ["jina pdf https://arxiv.org/pdf/2301.12345",
             "jina pdf 2301.12345",
             "jina pdf paper.pdf --type figure,table"],
        )

    try:
        result = api.extract_pdf(
            url=url, arxiv_id=arxiv_id, api_key=key,
            max_edge=max_edge, extract_type=extract_type,
        )
        click.echo(utils.format_pdf_results(result, as_json=as_json))
    except Exception as e:
        utils.handle_http_error(e)


# -- datetime --


@cli.command("datetime")
@click.argument("url", required=False)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def datetime_cmd(ctx, url, as_json):
    """Guess the publish/update date of a URL.

    \b
    Examples:
        jina datetime https://example.com/article
        echo "https://example.com" | jina datetime
    """
    if not url:
        stdin_lines = utils.read_stdin_lines()
        if stdin_lines:
            url = stdin_lines[0].strip()

    if not url:
        _short_usage(
            "Usage: jina datetime URL",
            ["jina datetime https://example.com/article",
             "echo URL | jina datetime"],
        )

    _validate_url(url)

    try:
        result = api.guess_datetime(url)
        if as_json:
            click.echo(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            data = result.get("data", result)
            if isinstance(data, dict):
                best = data.get("bestGuess", data.get("datetime", "unknown"))
                confidence = data.get("confidence", "?")
                click.echo(f"{best} (confidence: {confidence}%)")
            else:
                click.echo(str(data))
    except Exception as e:
        utils.handle_http_error(e)


# -- primer --


@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def primer(ctx, as_json):
    """Get context info - current time, location, network.

    No API key required.

    \b
    Examples:
        jina primer
        jina primer --json
    """
    try:
        result = api.primer()
        if as_json:
            click.echo(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            data = result.get("data", result)
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, dict):
                        click.echo(f"{k}:")
                        for k2, v2 in v.items():
                            click.echo(f"  {k2}: {v2}")
                    else:
                        click.echo(f"{k}: {v}")
            else:
                click.echo(str(data))
    except Exception as e:
        utils.handle_http_error(e)


# -- grep --


@cli.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    ),
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def grep(ctx, args):
    """Semantic grep using local Jina embeddings (requires jina-grep).

    Searches files semantically using natural language queries.
    Supports most GNU grep flags plus --threshold, --top-k, --model.

    \b
    Examples:
        jina grep "error handling" src/
        jina grep -r --threshold 0.3 "database connection" .
        grep -rn "error" src/ | jina grep "error handling logic"
        jina grep serve start

    \b
    Install: pip install jina-grep
    """
    if not args:
        _short_usage(
            "Usage: jina grep PATTERN [FILES...] [OPTIONS]",
            ['jina grep "error handling" src/',
             'jina grep -r "database connection" .',
             'grep -rn "error" src/ | jina grep "retry logic"',
             'jina grep serve start    (start local embedding server)'],
        )

    try:
        from jina_grep.cli import main as grep_main
    except ImportError:
        click.echo(
            "Error: jina-grep not installed.\n"
            "Fix: pip install jina-grep",
            err=True,
        )
        sys.exit(EXIT_USER_ERROR)

    # Replace sys.argv so jina-grep's CLI sees the right args
    import sys as _sys
    original_argv = _sys.argv
    _sys.argv = ["jina-grep"] + list(args)
    try:
        grep_main()
    except SystemExit as e:
        _sys.argv = original_argv
        sys.exit(e.code if e.code is not None else 0)
    finally:
        _sys.argv = original_argv


if __name__ == "__main__":
    cli()
