"""Print the real error behind 'Some scorer invocations failed during evaluation'.

    python mlflow/diagnose.py              # the most recent run in the experiment
    python mlflow/diagnose.py --run-id ... # a specific run

`DeepEvalScorer.__call__` catches every exception and returns `Feedback(error=e)`
instead of raising, so the harness only tells you *that* a scorer failed. The
message is stored on the trace as an assessment error. This pulls it out.

The three usual culprits, and what they look like here:

  ValueError: Failed to parse JSON output ...
      The judge wrapped its JSON in a ```json fence, or added a sentence around
      it. MLflow does structured output by prompt injection and then a bare
      json.loads -- no fence stripping. Fix: run with JUDGE_JSON_REPAIR=1
      (default in ragflow.py), or use a judge that obeys "JSON only".

  ... 400 / API key not valid / PERMISSION_DENIED ...
      GEMINI_API_KEY is wrong, expired, or is an OAuth token rather than an
      AI Studio API key. Get one at https://aistudio.google.com/apikey.

  ... 404 models/... is not found ...
      The judge model name is wrong for your key's API version. Try
      --judge gemini:/gemini-2.0-flash.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

for env_path in (PROJECT_ROOT / ".env", HERE / ".env"):
    if env_path.exists():
        load_dotenv(env_path)

import mlflow  # noqa: E402

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "enterprise-rag-eval")


def latest_run_id(experiment_name: str) -> tuple[str, str] | None:
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        print(f"No experiment named {experiment_name!r} on {TRACKING_URI}")
        return None
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
        output_format="list",
    )
    if runs is None or len(runs) == 0:
        print(f"No runs in {experiment_name!r} yet.")
        return None
    # Use getattr here because some MLflow type stubs incorrectly expose
    # ``info`` as a method, which makes direct ``.run_id`` access fail type
    # checking even though the runtime object is a Run.
    run_info = getattr(runs[0], "info", None)
    run_id = getattr(run_info, "run_id", None)
    if not run_id:
        print("The most recent run did not include a run ID.")
        return None
    return str(run_id), exp.experiment_id


def describe_error(err) -> str:
    """Assessment errors are an AssessmentError entity on modern MLflow and a
    plain string on older ones -- handle both."""
    if err is None:
        return ""
    code = getattr(err, "error_code", None)
    message = getattr(err, "error_message", None)
    if message or code:
        return f"[{code}] {message}".strip()
    return str(err)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--full", action="store_true", help="print whole error messages, not the first 4 lines"
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)

    experiment_id = None
    if args.run_id:
        run_id = args.run_id
        try:
            experiment_id = mlflow.get_run(run_id).info.experiment_id
        except Exception:
            pass
    else:
        found = latest_run_id(EXPERIMENT)
        if found is None:
            return 1
        run_id, experiment_id = found

    print(f"run: {run_id}   experiment: {EXPERIMENT}\n")

    # search_traces defaults to experiment 0 unless told otherwise, and errors
    # out if the run lives elsewhere -- pass the run's own experiment.
    kwargs = {"run_id": run_id, "return_type": "list"}
    if experiment_id is not None:
        kwargs["locations"] = [experiment_id]
    traces = mlflow.search_traces(**kwargs)
    if traces is None or len(traces) == 0:
        print("No traces on that run.")
        return 1

    failures = 0
    for trace in traces:
        # MLflow's type stubs currently infer items from ``search_traces`` as
        # Hashable, although the runtime values are Trace objects.
        trace = cast(Any, trace)
        assessments = getattr(trace.info, "assessments", None) or []
        request = str(trace.data.request or "")[:90]
        print(f"--- trace {trace.info.trace_id}  {request}")

        for assessment in assessments:
            error = getattr(assessment, "error", None)
            name = getattr(assessment, "name", "?")
            if error is None:
                value = getattr(assessment, "value", None)
                meta = getattr(assessment, "metadata", None) or {}
                print(f"    OK   {name}: {value}  score={meta.get('score')}")
                continue

            failures += 1
            text = describe_error(error)
            if not args.full:
                lines = text.splitlines()
                text = "\n         ".join(lines[:4])
                if len(lines) > 4:
                    text += "\n         ... (--full for the rest)"
            print(f"    FAIL {name}:\n         {text}")
        print()

    if failures:
        print(f"{failures} failed assessment(s). See the docstring at the top of this file.")
    else:
        print("No failed assessments on this run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
