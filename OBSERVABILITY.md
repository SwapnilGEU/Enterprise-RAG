# Observability

Three signals — traces, metrics, logs — from the API into Grafana, correlated
by `trace_id`, using **one** container.

```bash
docker compose up -d          # lgtm is already in the compose file
```

| | |
|---|---|
| Grafana | <http://localhost:3000> · admin / admin |
| Prometheus (raw) | <http://localhost:9090> |
| OTLP in | `4317` gRPC · `4318` HTTP |

Confirm it's live:

```bash
docker compose logs api | grep -E "tracing on|metrics on|log export on"
```

Three lines means all three signals are exporting. None means no OTLP endpoint
is configured, and the service runs exactly as it did before — telemetry is
never a reason it fails to start.

---

## Why one container

`grafana/otel-lgtm` bundles Prometheus, Tempo, Loki, Pyroscope, an
OpenTelemetry Collector and Grafana, with every datasource provisioned **and**
the three correlation links already wired:

- a log line → its trace (Loki `derivedFields`)
- a trace → that request's logs (Tempo `tracesToLogsV2`)
- a metric spike → the slow trace (Prometheus exemplars)

That is the entire config directory most tutorials have you build. It's already
inside the image, which is why this repo has almost no observability config —
only [`observability/`](observability/), which holds the dashboard.

> **A fair warning from Grafana's own docs:** this image is for development,
> demo and testing. Its sub-5-second startup was traded against scalability.
> A production deployment means separate Tempo / Loki / Prometheus services.

---

## The dashboard

Import [`observability/grafana/dashboards/enterprise-rag.json`](observability/grafana/dashboards/enterprise-rag.json):

**Grafana → Dashboards → New → Import → Upload JSON file**, then pick your
Prometheus and Loki datasources.

It covers request volume and rate, latency percentiles, **retrieval vs
generation time**, queue wait, degraded-answer rate, dependency health, and the
log stream.

---

## What gets collected

### Traces → Tempo

```
POST /query                  SERVER
├── POST <qdrant>            CLIENT   hybrid search + ColBERT rerank
└── POST <ollama>            CLIENT   generation
```

`/agent` has the same shape with more client spans — the pre-search, the
routing turn, any tool calls, and the final answer. Counting the Ollama spans
in one trace is the quickest way to see why an agent request was slow.

`/health` is excluded deliberately.

### Metrics → Prometheus

Request rate, latency percentiles, error rate and active requests come free
from the instrumentation. On top of that:

| metric | what it catches |
|---|---|
| `rag_generation_queue_wait_seconds` | time blocked on the 2-slot semaphore |
| `rag_generation_rejected_total` | 503s from the queue timeout |
| `rag_generation_active` | slots in use, 0–2 |
| `rag_answers_total{degraded=…}` | **degraded answers are HTTP 200** — invisible to error rate |
| `rag_retrieval_sources` | zero-source answers: fast, successful, useless |
| `rag_agent_tool_calls_total{tool=…}` | which tool the model actually chose |
| `rag_component_ready{component=…}` | qdrant / ollama / agent up or down |

Histogram buckets reach 300 seconds. The SDK default stops at 10, which would
put every LLM generation in the overflow bucket and make percentiles fiction.

### Logs → Loki

Every line carries `request_id`, and `trace_id` when inside a request — which
is what makes a log line clickable through to its trace.

```logql
{service_name="enterprise-rag-api"}
{service_name="enterprise-rag-api"} | event="query_answered"
{service_name="enterprise-rag-api"} | event="refused"
{service_name="enterprise-rag-api"} | degraded="true"
{service_name="enterprise-rag-api"} | duration_ms > 10000
```

> **Don't use `| json`.** Logs arrive over OTLP, so the line body is the
> message and every field is Loki *structured metadata* — filter it directly,
> as above. `| json` tries to parse `warmup finished` as an object and fails
> with `JSONParserErr`. (The JSON formatter shapes **stdout** only, which is
> what `docker logs` shows.)

`/health` and `/ready` are logged at DEBUG. A healthcheck and a UI sidebar
polling every few seconds would otherwise be most of what Loki stores — set
`LOG_LEVEL=DEBUG` when you want them.

---

## MLflow is a different thing

Worth being clear, because both involve traces:

| question | tool |
|---|---|
| Is the server up? Why was that request slow? | **Grafana / LGTM** |
| Did retrieval improve when I added documents? | **MLflow** |
| Is the answer *correct*? | **MLflow** — Prometheus can't tell you an answer was wrong |

They stay separate on purpose. The serving path exports to Tempo and disables
MLflow tracing; eval runs do the opposite. If an eval ever reports every scorer
failing, check that `OTEL_EXPORTER_OTLP_ENDPOINT` isn't set in its environment
— MLflow treats a configured OTLP endpoint as *exclusive* and will send the
trace to the collector instead of to MLflow, leaving the scorers with nothing
to read.

That's why telemetry config lives in `api/.env` (loaded only by the API) and
never in the repo-root `.env` (loaded by the eval flows too).

---

## Configuration

Under Docker, `docker-compose.yml` sets these on the `api` service. Running
locally, copy `api/.env.example` to `api/.env`.

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317   # http://lgtm:4317 in compose
OTEL_SERVICE_NAME=enterprise-rag-api
OTEL_METRIC_EXPORT_INTERVAL=15000
```

Use the **general** endpoint variable, not `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`.
Each exporter reads its own variables, and the metric and log exporters never
look at the traces one — they'd fall back to `localhost:4317`, which inside a
container is the container itself. Traces would arrive and metrics and logs
would vanish with no error anywhere.

Unset it entirely and telemetry is simply off.

---

## Troubleshooting

**Panels empty after a laptop sleep.** Prometheus is the only one of the three
backends that validates timestamps — it silently rejects samples that are out
of order or in the future. A clock that drifted while the machine was suspended
breaks metrics and nothing else. Compare `docker compose exec lgtm date` with
your own clock; a full Docker Desktop restart resyncs the VM.

**Panels empty after a restart, generally.** Counters reset to zero and
`rate()` needs two samples. Send a few requests, then widen the time range.

**A panel says "No data".** Usually a metric-name mismatch. The dashboard
assumes the legacy HTTP conventions (`http_server_duration_milliseconds_*`),
which is what the instrumentation emits by default. Check Prometheus' metric
browser before editing queries.

**Nothing at all in Prometheus.** Query `target_info` — it's created
automatically for every resource. Present means ingestion works and it's a
naming problem; absent means nothing is landing.
