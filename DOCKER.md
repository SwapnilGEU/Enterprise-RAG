# Running with Docker

## Needs to be running on Windows first

| | check |
|---|---|
| Docker Desktop | `docker ps` |
| Ollama, with qwen3:4b-instruct | `curl http://localhost:11434/api/tags` |
| Postgres on port 5432 | `psql -U postgres -d RAG -c "select 1"` |
| `.env` in this folder | must have QDRANT_URL, QDRANT_API_KEY, PG_PASSWORD |

Qdrant is cloud — nothing to start.

## Build (once, and after any code change)

```
docker compose build
```

## Run

```
docker compose up -d
```

Open http://localhost:8501

## Check it works

```
curl http://localhost:8000/ready
```

All three (qdrant, ollama, agent) should say `"ready":true`.
The first check can take ~30s while Ollama loads the model.

## Watch logs

```
docker compose logs -f api
```

## Stop

```
docker compose down
```

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
