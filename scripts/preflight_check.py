from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


MODEL_IDS = ["simgnn", "multiscale-set", "segmn", "graph-fusion", "graph2region"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Meeting preflight check for the graph similarity Flask app.")
    parser.add_argument("--base-url", default="http://127.0.0.1:5002")
    args = parser.parse_args()

    checks = [
        ("home", lambda: request(args.base_url + "/")),
        ("datasets", lambda: request_json(args.base_url + "/api/datasets")),
        ("graphs", lambda: request_json(args.base_url + "/api/datasets/aids700nef/graphs")),
        (
            "exact-best-pair",
            lambda: request_json(
                args.base_url + "/api/datasets/aids700nef/best-pair",
                {"methods": ["exact-ged"], "max_pairs": 3, "scope": "train-test"},
            ),
        ),
        (
            "all-model-compare",
            lambda: compare_first_pair(args.base_url, MODEL_IDS),
        ),
        (
            "all-model-benchmark",
            lambda: request_json(
                args.base_url + "/api/datasets/aids700nef/evaluate",
                {
                    "methods": MODEL_IDS,
                    "sample_size": 2,
                    "sample_mode": "stratified",
                    "scope": "train-test",
                    "seed": 379,
                    "top_k": 1,
                },
            ),
        ),
        (
            "training-plans",
            lambda: request_json(args.base_url + "/api/training?dataset=aids700nef"),
        ),
        (
            "research-summary",
            lambda: request_json(args.base_url + "/api/research-summary"),
        ),
    ]

    failures = 0
    for name, check in checks:
        try:
            payload = check()
            print(f"OK {name}: {summarize(name, payload)}")
        except Exception as exc:  # noqa: BLE001 - this is a small CLI health check.
            failures += 1
            print(f"FAIL {name}: {exc}")

    return 1 if failures else 0


def request(url: str) -> str:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read().decode("utf-8", errors="replace")


def request_json(url: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request_obj = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if payload is not None else {},
        method="POST" if payload is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request_obj, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def compare_first_pair(base_url: str, methods: list[str]) -> dict:
    pair = request_json(base_url + "/api/datasets/aids700nef")
    return request_json(
        base_url + "/api/compare",
        {
            "dataset": "aids700nef",
            "left": pair["left"],
            "right": pair["right"],
            "meta": pair["meta"],
            "methods": methods,
        },
    )


def summarize(name: str, payload: object) -> str:
    if name == "home":
        return "HTML loaded"
    if name == "datasets" and isinstance(payload, dict):
        return f"{len(payload.get('datasets', []))} datasets"
    if name == "graphs" and isinstance(payload, dict):
        return f"{len(payload.get('train', []))} train / {len(payload.get('test', []))} test"
    if name == "exact-best-pair" and isinstance(payload, dict):
        winner = payload["search"]["winner"]
        return f"{winner['left_graph']} vs {winner['right_graph']} GED {winner['exact_ged']}"
    if name == "all-model-compare" and isinstance(payload, dict):
        statuses = {
            result["id"]: result["status"]
            for result in payload.get("results", [])
        }
        if set(statuses) != set(MODEL_IDS) or any(
            status != "executed" for status in statuses.values()
        ):
            raise RuntimeError(f"Not every model executed: {statuses}")
        return f"{len(statuses)}/{len(MODEL_IDS)} local checkpoints executed"
    if name == "all-model-benchmark" and isinstance(payload, dict):
        statuses = {
            result["id"]: result["status"]
            for result in payload.get("models", [])
        }
        if set(statuses) != set(MODEL_IDS) or any(
            status != "evaluated" for status in statuses.values()
        ):
            raise RuntimeError(f"Not every model evaluated: {statuses}")
        return f"{len(statuses)}/{len(MODEL_IDS)} models evaluated; artifact={payload.get('artifact_path')}"
    if name == "training-plans" and isinstance(payload, dict):
        plans = payload.get("plans", [])
        runnable = sum(bool(plan.get("can_start")) for plan in plans)
        if runnable != len(MODEL_IDS):
            raise RuntimeError(f"Only {runnable}/{len(MODEL_IDS)} training plans can start")
        return f"{runnable}/{len(MODEL_IDS)} training plans ready"
    if name == "research-summary" and isinstance(payload, dict):
        audit = payload.get("checkpoint_audit") or {}
        return f"checkpoint audit {audit.get('verified', 0)}/{audit.get('total', 0)}"
    return "loaded"


if __name__ == "__main__":
    sys.exit(main())
