# Running with Docker

## Needs to be running on Windows first

| | check |
|---|---|
| Docker Desktop | `docker ps` |
| Ollama, with qwen3:4b-instruct | `curl http://localhost:11434/api/tags` |
| Postgres on port 5432 | `psql -U postgres -d RAG -c "select 1"` |
| `.env` in this folder | must have QDRANT_URL, QDRANT_API_KEY, PG_PASSWORD |

Qdrant is cloud — nothing to start.

## What comes up

| service | image | port |
|---|---|---|
| `api` | built here | 8000 |
| `ui` | built here | 8501 |
| `lgtm` | `grafana/otel-lgtm` | 3000 Grafana · 4317/4318 OTLP · 9090 Prometheus |

`lgtm` is pulled, not built, so `docker compose build` skips it. The **first**
`up` downloads about 1GB — it looks stalled for a minute and isn't.

## Build (once, and after any code change)

```
docker compose build
```

## Run

```
docker compose up -d
```

Or in one step: `docker compose up -d --build`.

| | |
|---|---|
| Ask questions | http://localhost:8501 |
| API docs | http://localhost:8000/docs |
| Dashboards | http://localhost:3000 — admin / admin |

## Check it works

```
curl http://localhost:8000/ready
```

All three (qdrant, ollama, agent) should say `"ready":true`.
The first check can take ~30s while Ollama loads the model.

Telemetry should report itself at boot:

```
docker compose logs api | grep -E "tracing on|metrics on|log export on"
```

Three lines means traces, metrics and logs are all exporting. See
[OBSERVABILITY.md](OBSERVABILITY.md).

## Watch logs

```
docker compose logs -f api
```

## Stop

```
docker compose down
```

Telemetry survives this — `lgtm` writes to the `lgtm-data` volume. To throw it
away as well, `docker compose down -v`.

---

## Two terminals instead of compose

Same images. Run `docker compose build` first, then:

Terminal 1:
```
docker run -p 8000:8000 --env-file .env -e PG_HOST=host.docker.internal -e OLLAMA_BASE_URL=http://host.docker.internal:11434 rag-api
```

Terminal 2:
```
docker run -p 8501:8501 rag-ui
```

The two `-e` flags are needed because `--env-file .env` overrides the
Dockerfile's ENV, and `.env` says `localhost` — which inside a container
means the container itself, not Windows.

---

## If something fails

Check `/ready` first — it names the broken component and the error.

| component down | cause |
|---|---|
| qdrant | QDRANT_URL / QDRANT_API_KEY wrong or missing in `.env` |
| ollama | Ollama not running on Windows |
| agent | Postgres not running, or PG_PASSWORD wrong |

`missing_settings` non-empty means `.env` wasn't picked up at all — it must
be in the same folder as docker-compose.yml.

**UI changes not showing?** `ui/app.py` is baked into the image at build time.
`docker compose build ui` (or `up -d --build`) is required.

**API can't reach Ollama?** Read the errno before reaching for the usual
answer. `ECONNREFUSED` means it routed fine and nothing was listening — that is
the `OLLAMA_HOST=0.0.0.0` case. `ENETUNREACH` means it couldn't route at all,
which points at the `extra_hosts` line in `docker-compose.yml`: Docker Desktop
already provides `host.docker.internal`, and overriding it with the bridge
gateway resolves to an address with no route to Windows. The line is only
needed on native Linux. It is also a container-creation setting, so
`--force-recreate` is required for a change to it to take effect.

No restart needed after fixing a dependency: `/ready` retries every 10s.

---

## Push to Docker Hub

```
docker login
docker tag rag-api YOURNAME/rag-api:0.1.0
docker tag rag-ui  YOURNAME/rag-ui:0.1.0
docker push YOURNAME/rag-api:0.1.0
docker push YOURNAME/rag-ui:0.1.0
```

`.env` is not in the images (see `.dockerignore`), so whoever runs them
supplies their own with `--env-file`.
