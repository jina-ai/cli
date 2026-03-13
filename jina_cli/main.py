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
"""

import sys
import json

import click

from jina_cli import __version__
from jina_cli import api, utils


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


@click.group(cls=AliasedGroup)
@click.version_option(__version__, prog_name="jina")
@click.option("--api-key", envvar="JINA_API_KEY", default=None, hidden=True,
              help="Jina API key (or set JINA_API_KEY env var)")
@click.pass_context
def cli(ctx, api_key):
    """Jina AI CLI - search, read, embed, rerank, and more.

    All Jina AI APIs as Unix-friendly commands. Supports pipes and chaining.

    Set your API key: export JINA_API_KEY=your-key
    Get a key: https://jina.ai/?sui=apikey
    """
    ctx.ensure_object(dict)
    ctx.obj["api_key"] = api_key


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
        click.echo(click.get_current_context().get_help())
        sys.exit(1)

    try:
        for u in urls:
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
        click.echo(click.get_current_context().get_help())
        sys.exit(1)

    key = api_key or ctx.obj.get("api_key")
    tbs_val = f"qdr:{tbs}" if tbs else None

    try:
        if blog:
            result = api.search_blog(query, api_key=key, num=num, tbs=tbs_val, as_json=as_json)
            if as_json and isinstance(result, dict):
                click.echo(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                click.echo(result)
        elif arxiv:
            result = api.search_arxiv(query, api_key=key, num=num, tbs=tbs_val, as_json=as_json)
            if as_json and isinstance(result, dict):
                click.echo(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                click.echo(result)
        elif ssrn:
            result = api.search_ssrn(query, api_key=key, num=num, tbs=tbs_val, as_json=as_json)
            if as_json and isinstance(result, dict):
                click.echo(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                click.echo(result)
        elif images:
            result = api.search_images(query, api_key=key, num=num, tbs=tbs_val, gl=gl, hl=hl, as_json=as_json)
            if as_json and isinstance(result, dict):
                click.echo(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                click.echo(result)
        else:
            result = api.search_web(
                query, api_key=key, num=num, tbs=tbs_val,
                location=location, gl=gl, hl=hl, as_json=as_json,
            )
            if as_json and isinstance(result, dict):
                click.echo(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                click.echo(result)
    except Exception as e:
        utils.handle_http_error(e)


# -- embed --


@cli.command()
@click.argument("text", nargs=-1)
@click.option("--model", default="jina-embeddings-v3", help="Model name (default: jina-embeddings-v3)")
@click.option("--task", default="text-matching",
              type=click.Choice(["retrieval.query", "retrieval.passage", "text-matching",
                                 "classification", "separation"]),
              help="Embedding task type")
@click.option("--dimensions", type=int, default=None, help="Output dimensions (Matryoshka)")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--api-key", default=None, help="Jina API key")
@click.pass_context
def embed(ctx, text, model, task, dimensions, as_json, api_key):
    """Generate embeddings for text.

    Input from arguments or stdin (one text per line).

    \b
    Examples:
        jina embed "hello world"
        echo "hello world" | jina embed
        jina embed "text1" "text2" "text3"
        cat texts.txt | jina embed --json
    """
    key = api_key or ctx.obj.get("api_key")

    texts = list(text)
    if not texts:
        stdin_lines = utils.read_stdin_lines()
        texts = stdin_lines

    if not texts:
        click.echo(click.get_current_context().get_help())
        sys.exit(1)

    try:
        result = api.embed(texts, api_key=key, model=model, task=task, dimensions=dimensions)
        click.echo(utils.format_embeddings(result, as_json=as_json))
    except Exception as e:
        utils.handle_http_error(e)


# -- rerank --


@cli.command()
@click.argument("query")
@click.option("-n", "--top-n", type=int, default=None, help="Max results to return")
@click.option("--model", default="jina-reranker-v2-base-multilingual", help="Reranker model")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--api-key", default=None, help="Jina API key")
@click.pass_context
def rerank(ctx, query, top_n, model, as_json, api_key):
    """Rerank documents by relevance to a query.

    Reads documents from stdin, one per line.

    \b
    Examples:
        jina search "AI" | jina rerank "embeddings"
        cat docs.txt | jina rerank "machine learning"
        echo -e "doc1\\ndoc2\\ndoc3" | jina rerank "query" --top-n 2
    """
    key = api_key or ctx.obj.get("api_key")
    documents = utils.read_stdin_lines()

    if not documents:
        click.echo("Error: no documents on stdin. Pipe text to rerank.", err=True)
        click.echo("Example: cat docs.txt | jina rerank \"your query\"", err=True)
        sys.exit(1)

    try:
        result = api.rerank(query, documents, api_key=key, model=model, top_n=top_n)
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
        click.echo("Error: no input on stdin. Pipe text to deduplicate.", err=True)
        click.echo("Example: cat items.txt | jina dedup", err=True)
        sys.exit(1)

    try:
        result = api.deduplicate(lines, api_key=key, k=k)
        click.echo(utils.format_dedup_results(result, as_json=as_json))
    except Exception as e:
        utils.handle_http_error(e)


# -- screenshot --


@cli.command()
@click.argument("url", required=False)
@click.option("--full-page", is_flag=True, help="Capture full page (not just viewport)")
@click.option("-o", "--output", default=None, help="Save to file (default: stdout as base64)")
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
        click.echo(click.get_current_context().get_help())
        sys.exit(1)

    try:
        result = api.screenshot_url(url, api_key=key, full_page=full_page)
        if as_json:
            click.echo(json.dumps(result, indent=2, ensure_ascii=False))
        elif output:
            # Extract image data and save
            data = result.get("data", result)
            if isinstance(data, dict):
                img_data = data.get("screenshot", data.get("image", ""))
            else:
                img_data = str(data)
            if img_data.startswith("data:"):
                # Strip data URI prefix
                img_data = img_data.split(",", 1)[1] if "," in img_data else img_data
            import base64
            with open(output, "wb") as f:
                f.write(base64.b64decode(img_data))
            click.echo(f"Saved to {output}", err=True)
        else:
            click.echo(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        utils.handle_http_error(e)


# -- bibtex --


@cli.command()
@click.argument("query")
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
    key = api_key or ctx.obj.get("api_key")

    try:
        results = api.search_bibtex(query, api_key=key, author=author, year=year, num=num)
        click.echo(utils.format_bibtex_results(results, as_json=as_json))
    except Exception as e:
        utils.handle_http_error(e)


# -- expand --


@cli.command()
@click.argument("query")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--api-key", default=None, help="Jina API key")
@click.pass_context
def expand(ctx, query, as_json, api_key):
    """Expand a search query into related queries.

    \b
    Examples:
        jina expand "machine learning"
        jina expand "climate change effects" --json
    """
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
        click.echo(click.get_current_context().get_help())
        sys.exit(1)

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
        click.echo(click.get_current_context().get_help())
        sys.exit(1)

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


if __name__ == "__main__":
    cli()
