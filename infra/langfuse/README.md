# Local Langfuse — shared tracing backend

A self-hosted [Langfuse](https://langfuse.com) instance for **any project on this
machine**. Run it once; point every project at `http://localhost:3100` with its
own project API keys. Think "Jaeger for LLM apps" — a trace/span backend with a
UI, plus LLM-specific extras (token cost, model, prompt/completion I/O, eval
scores).

This folder is **self-contained and portable** — to truly share it across repos,
copy it somewhere neutral (e.g. `C:\llm\langfuse`) and run it from there. Data
lives in named Docker volumes, independent of which project connects.

## What runs (≈6 containers, only the UI is exposed)

`langfuse-web` (UI/API, **the only published port: 3000**) + `langfuse-worker`
(async ingestion) + Postgres (transactional) + ClickHouse (trace analytics) +
Redis (queue) + MinIO (blob storage). Postgres/ClickHouse/Redis/MinIO are
internal-only, so this never clashes with a project's own Postgres.

## Start

```bash
cd infra/langfuse          # or wherever you copied this folder
docker compose up -d       # first boot ~1 min (ClickHouse runs migrations)
```

Open http://localhost:3100 and log in with the seeded user:

- **email:** `admin@local.test`   **password:** `langfuse-local`

The `rag-historian` project already exists with fixed API keys (see `.env`).

> The `LANGFUSE_INIT_*` provisioning in `.env` only applies on an **empty DB**
> (first boot). Changing those values later does nothing unless you wipe volumes.

## Connect a project

In that project's `.env` (this repo's already done — see below):

```
LANGFUSE_BASE_URL=http://localhost:3100
LANGFUSE_PUBLIC_KEY=pk-lf-...     # from infra/langfuse/.env (or the UI)
LANGFUSE_SECRET_KEY=sk-lf-...
```

## Add another project (for a different repo)

1. In the UI: **+ New project** → name it.
2. Project **Settings → API keys → Create** → copy the public + secret keys.
3. Put those keys + `LANGFUSE_BASE_URL=http://localhost:3100` in that repo's `.env`.

One instance, many projects, traces kept separate per project.

## Stop / wipe

```bash
docker compose down        # stop; data persists in volumes
docker compose down -v     # stop and DELETE all traces/data
```

## Notes

- **Secrets** (`.env`) are local-dev only — the instance binds to `127.0.0.1`, so
  nothing is exposed off-machine. `.env` is gitignored; `.env.example` shows how to
  regenerate.
- **vs. OpenTelemetry/Jaeger:** Langfuse's `trace → generation/span` maps onto OTel
  `trace → span`, but it's LLM-specialized (cost/tokens/scores/prompt-mgmt) and the
  SDK uses Langfuse's own ingestion protocol (it also exposes an OTLP endpoint if
  you prefer OTel exporters).
