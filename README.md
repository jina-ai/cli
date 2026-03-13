# jina-cli

All Jina AI APIs as Unix commands. Search, read, embed, rerank - with pipes.

This CLI is designed for both humans and AI agents. An agent with shell access needs only `run(command="jina search ...")` instead of managing 20 separate tool definitions. The CLI supports pipes, chaining (`&&`, `||`, `;`), and `--help` for self-discovery.

## Install

```bash
pip install jina-cli
# or
uv pip install jina-cli
```

Set your API key:
```bash
export JINA_API_KEY=your-key-here
# Get one at https://jina.ai/?sui=apikey
```

## Commands

| Command | Description |
|---------|-------------|
| `jina read URL` | Extract clean markdown from web pages |
| `jina search QUERY` | Web search (also --arxiv, --ssrn, --images, --blog) |
| `jina embed TEXT` | Generate embeddings |
| `jina rerank QUERY` | Rerank documents from stdin by relevance |
| `jina dedup` | Deduplicate text from stdin |
| `jina screenshot URL` | Capture screenshot of a URL |
| `jina bibtex QUERY` | Search BibTeX citations (DBLP + Semantic Scholar) |
| `jina expand QUERY` | Expand a query into related queries |
| `jina pdf URL` | Extract figures/tables/equations from PDFs |
| `jina datetime URL` | Guess publish/update date of a URL |
| `jina primer` | Context info (time, location, network) |
| `jina grep PATTERN` | Semantic grep (requires `pip install jina-grep`) |

## Pipes

The point of a CLI is composability. Every command reads from stdin and writes to stdout.

```bash
# Search and rerank
jina search "transformer models" | jina rerank "efficient inference"

# Read multiple URLs
cat urls.txt | jina read

# Search, deduplicate results
jina search "attention mechanism" | jina dedup

# Chain searches
jina expand "climate change" | head -1 | xargs -I {} jina search "{}"

# Get BibTeX for arXiv results
jina search --arxiv "BERT" --json | jq -r '.results[].title' | head -3
```

## Usage

### Read web pages

```bash
jina read https://example.com
jina read https://example.com --links --images
echo "https://example.com" | jina read
```

### Search

```bash
jina search "what is BERT"
jina search --arxiv "attention mechanism" -n 10
jina search --ssrn "corporate governance"
jina search --images "neural network diagram"
jina search --blog "embeddings"
jina search "AI news" --time d          # past day
jina search "LLMs" --gl us --hl en     # US, English
```

### Embed

```bash
jina embed "hello world"
jina embed "text1" "text2" "text3"
cat texts.txt | jina embed
jina embed "hello" --model jina-embeddings-v3 --task retrieval.query
```

### Rerank

```bash
cat docs.txt | jina rerank "machine learning"
jina search "AI" | jina rerank "embeddings" --top-n 5
```

### Deduplicate

```bash
cat items.txt | jina dedup
cat items.txt | jina dedup -k 10
```

### Screenshot

```bash
jina screenshot https://example.com                        # prints screenshot URL
jina screenshot https://example.com -o page.png            # saves to file
jina screenshot https://example.com --full-page -o page.jpg
```

### BibTeX

```bash
jina bibtex "attention is all you need"
jina bibtex "transformer" --author Vaswani --year 2017
```

### PDF extraction

```bash
jina pdf https://arxiv.org/pdf/2301.12345
jina pdf 2301.12345                        # arXiv ID shorthand
jina pdf https://example.com/paper.pdf --type figure,table
```

## JSON output

Every command supports `--json` for structured output, useful for piping to `jq`:

```bash
jina search "BERT" --json | jq '.results[0].url'
jina read https://example.com --json | jq '.data.content'
```

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | User/input error (missing args, bad input, missing API key) |
| 2 | API/server error (network, timeout, server error) |
| 130 | Interrupted (Ctrl+C) |

Useful for scripting and agent workflows:

```bash
jina search "query" && echo "success" || echo "failed with $?"
```

## Environment variables

| Variable | Description |
|----------|-------------|
| `JINA_API_KEY` | API key for Jina services (required for most commands) |

## For AI agents

An agent with shell access can use this CLI directly:

```python
result = run(command="jina search 'transformer architecture'")
result = run(command="jina read https://arxiv.org/abs/2301.12345")
result = run(command="jina search 'AI' | jina rerank 'embeddings'")
```

No tool catalog needed. The agent discovers capabilities via `jina --help` and `jina search --help`. Errors include actionable guidance.

## Semantic grep

`jina grep` provides semantic search over files using local [Jina embeddings on MLX](https://github.com/jina-ai/jina-grep-cli). It requires a separate install:

```bash
pip install jina-grep
```

```bash
jina grep "error handling" src/
jina grep -r --threshold 0.3 "database connection" .
grep -rn "error" src/ | jina grep "retry logic"
```

Supports most GNU grep flags (`-r`, `-n`, `-l`, `-c`, `-A/-B/-C`, `--include`, `--exclude`) plus semantic flags (`--threshold`, `--top-k`, `--model`). Run `jina grep --help` for full options.

### Server mode

For repeated queries, start a persistent embedding server to avoid model reload:

```bash
jina grep serve start    # background server, model stays in GPU memory
jina grep serve stop     # stop when done
```

## Local mode

`jina embed` and `jina rerank` support `--local` to run on Apple Silicon via the jina-grep embedding server instead of the Jina API. No API key needed.

```bash
# Start the local server first
jina grep serve start

# Local embeddings
jina embed --local "hello world"
cat texts.txt | jina embed --local --json

# Local reranking (cosine similarity on local embeddings)
cat docs.txt | jina rerank --local "machine learning"
```

Local mode uses `jina-embeddings-v5-nano` by default. Override with `--model jina-embeddings-v5-small`.

Requires `pip install jina-grep` and `jina grep serve start`.

## License

Apache-2.0
