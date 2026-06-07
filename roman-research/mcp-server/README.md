# Roman Research — MCP Server

Exposes the Caesar corpus over the [Model Context Protocol](https://modelcontextprotocol.io)
so any MCP host (Claude Code, Claude Desktop, Cursor) can research it. It wraps
the **same retrieval stack** the Module 7 agent uses (`contextual-v1` chunks +
rerank-on-context) — a second front door to `query/retrieve.ts`, not a new
retriever.

## What it exposes

The three MCP primitives, each on a different control axis:

| Primitive | Control | Name | Purpose |
|---|---|---|---|
| **Tool** | model | `search_roman_corpus(query, top_k?)` | semantic search, all four sources |
| **Tool** | model | `search_roman_source(source, query, top_k?)` | search one source in isolation (contradiction handling) |
| **Tool** | model | `read_roman_chunk(chunk_id)` | full text of a chunk |
| **Tool** | model | `list_roman_sources()` | books + chapter/chunk counts |
| **Tool** | model | `cite_passage(chunk_id)` | formatted citation string |
| **Resource** | app | `roman://sources` | corpus index |
| **Resource** | app | `roman://source/{slug}` | one book: metadata + chapter listing (4 enumerable) |
| **Prompt** | user | `research_topic(topic)` | multi-source cited research workflow |
| **Prompt** | user | `summarize_event(event)` | summarize an event, agreements vs. differences |

## Prerequisites

The server connects to live services on startup — bring them up first:

- **ParadeDB** (Postgres): `docker compose up -d`, with `DATABASE_URL` in the repo-root `.env`
- **llama-swap** on `:8080` serving `bge-m3` (embeddings) + `bge-reranker-v2-m3` (rerank) — `C:\llm`
- Corpus ingested at `contextual-v1` (`pnpm dev roman-research/ingest/index.ts --version=contextual-v1`)

## Run / wire into a host

It speaks JSON-RPC over **stdio** — the host spawns it. Run directly to sanity-check:

```bash
pnpm dev roman-research/mcp-server/index.ts
# → stderr: "[roman-research mcp] connected over stdio; corpus ready."
```

### Claude Code

Project config is already checked in at repo-root [`.mcp.json`](../../.mcp.json):

```json
{
  "mcpServers": {
    "roman-research": {
      "command": "pnpm",
      "args": ["dev", "roman-research/mcp-server/index.ts"]
    }
  }
}
```

Claude Code picks it up on the next session start in this repo; approve the
project server when prompted, then `/mcp` lists it. The `research_topic` /
`summarize_event` prompts appear as slash commands.

> **Windows note:** if the host fails to spawn `pnpm` (a `.cmd` shim), use
> `"command": "cmd", "args": ["/c", "pnpm", "dev", "roman-research/mcp-server/index.ts"]`.

### Claude Desktop

Add the same block to `claude_desktop_config.json` (Settings → Developer → Edit
Config), using an **absolute** path in the args and a `"cwd"` pointing at the
repo root so `.env` and relative imports resolve. Restart Claude Desktop.

## Debugging

Use the [MCP Inspector](https://github.com/modelcontextprotocol/inspector) —
the only sane way to poke the server without a full host:

```bash
npx @modelcontextprotocol/inspector pnpm dev roman-research/mcp-server/index.ts
```

**Footgun:** stdout is the protocol channel. Anything printed there (a stray
`console.log`) corrupts the JSON-RPC framing and the host silently drops the
server. All diagnostics in this server go to **stderr** — keep it that way.
