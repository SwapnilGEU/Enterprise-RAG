# observability/

The **only** thing that needs to live on disk for the LGTM stack.

`grafana/otel-lgtm` already ships Prometheus, Tempo, Loki, Pyroscope, a
Collector and Grafana, with all four datasources provisioned *and* the three
correlation links wired (Loki `derivedFields` → Tempo, Tempo `tracesToLogsV2`
→ Loki, Prometheus `exemplarTraceIdDestinations` → Tempo). So there is no
`tempo.yml`, no `prometheus.yml`, no datasource provisioning here — copying
those from a reference repo would just duplicate what the image has.

Custom dashboards are the exception: they are ours, and they should be in git.

## dashboards/enterprise-rag.json

Import it: Grafana (`http://localhost:3000`, admin/admin) → Dashboards →
New → Import → **Upload JSON file** → pick this file → choose the Prometheus
and Loki datasources when prompted.

It is not auto-provisioned. Mounting it into the container would mean
overriding the image's own provisioning directory, which risks clobbering the
correlation config above. Importing by hand once is cheaper than that risk.

### If a panel says "No data"

Almost always a metric-name mismatch. The names in this dashboard assume the
**legacy** HTTP semantic conventions, which is what the instrumentation emits
by default:

    http_server_duration_milliseconds_{bucket,count,sum}
    http_client_duration_milliseconds_{bucket,count,sum}   labelled net_peer_name
    http_server_active_requests

Setting `OTEL_SEMCONV_STABILITY_OPT_IN=http` would switch them to
`http_server_request_duration_seconds_*` with a `server_address` label — and
every HTTP panel here would need updating (and the `/1000` conversions
removed). Check what you actually have in Prometheus' metric browser at
`http://localhost:9090` before editing queries.

The `rag_*` metrics are ours and their names are stable either way — see
`claude/telemetry-inventory.md` for the full list.
