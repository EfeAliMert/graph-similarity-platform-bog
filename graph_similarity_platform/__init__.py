from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request

from .data import (
    DatasetUploadError,
    MAX_UPLOAD_BYTES,
    list_original_datasets,
    list_original_graphs,
    load_original_dataset,
    load_original_pair,
    local_archives,
    original_pair_matches_graphs,
    pair_ground_truth,
    sample_pair,
    save_uploaded_dataset,
)
from .evaluation import benchmark_catalog, evaluate_models, load_benchmark
from .graph_utils import GraphInputError, graph_from_payload
from .hpo.service import optimization_catalog
from .models.real_models import MODELS, run_models
from .model_runs import model_run_job, start_model_run
from .research_summary import (
    checkpoint_audit_summary,
    latest_research_matrix_status,
    latest_research_summary,
)
from .search import (
    BestPairSearchError,
    evaluate_prefilter_ablation,
    find_best_pair,
    reranking_job,
    start_reranking_ablation,
)
from .training import start_training, training_catalog


BASE_DIR = Path(__file__).resolve().parent.parent


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(BASE_DIR / "templates"),
        static_folder=str(BASE_DIR / "static"),
    )
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

    @app.errorhandler(413)
    def request_too_large(_error):
        return jsonify({"error": "Upload exceeds the 512 MB request limit."}), 413

    @app.get("/")
    def index():
        default_pair = sample_pair()
        asset_version = _asset_version()
        return render_template(
            "index.html",
            methods=MODELS,
            archives=local_archives(),
            datasets=list_original_datasets(),
            default_left=json.dumps(default_pair["left"], indent=2),
            default_right=json.dumps(default_pair["right"], indent=2),
            default_meta=default_pair.get("meta", {}),
            asset_version=asset_version,
        )

    @app.get("/favicon.ico")
    def favicon():
        return Response(status=204)

    @app.get("/api/datasets")
    def api_datasets():
        return jsonify({"datasets": list_original_datasets()})

    @app.post("/api/datasets/upload")
    def api_dataset_upload():
        try:
            dataset = save_uploaded_dataset(
                archive_file=request.files.get("archive"),
                ground_truth_file=request.files.get("ground_truth"),
                name=request.form.get("name", ""),
                dataset_id=request.form.get("dataset_id"),
                domain=request.form.get("domain"),
            )
            return jsonify({"dataset": dataset}), 201
        except (DatasetUploadError, OSError, ValueError, json.JSONDecodeError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/datasets/<dataset_id>")
    def api_dataset(dataset_id: str):
        try:
            return jsonify(load_original_dataset(dataset_id))
        except (FileNotFoundError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 404

    @app.get("/api/datasets/<dataset_id>/graphs")
    def api_dataset_graphs(dataset_id: str):
        try:
            return jsonify(list_original_graphs(dataset_id))
        except (FileNotFoundError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 404

    @app.get("/api/datasets/<dataset_id>/pair")
    def api_dataset_pair(dataset_id: str):
        left = request.args.get("left")
        right = request.args.get("right")
        if not left or not right:
            return jsonify({"error": "Pair request must include left and right graph members."}), 400
        try:
            return jsonify(load_original_pair(dataset_id, left, right))
        except (FileNotFoundError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 404

    @app.post("/api/datasets/<dataset_id>/best-pair")
    def api_dataset_best_pair(dataset_id: str):
        payload = request.get_json(silent=True) or {}
        method_ids = payload.get("methods") or ["simgnn"]
        max_pairs = payload.get("max_pairs", 8)
        scope = payload.get("scope", "train-test")
        try:
            return jsonify(find_best_pair(dataset_id, method_ids, max_pairs=max_pairs, scope=scope))
        except BestPairSearchError as exc:
            return jsonify({"error": str(exc), "search": exc.payload}), exc.status_code
        except (FileNotFoundError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/datasets/<dataset_id>/retrieval-ablation")
    def api_dataset_retrieval_ablation(dataset_id: str):
        payload = request.get_json(silent=True) or {}
        try:
            return jsonify(
                evaluate_prefilter_ablation(
                    dataset_id,
                    payload.get("budgets") or [1, 4, 8, 16, 32, 64],
                    scope=payload.get("scope", "train-test"),
                    top_k=payload.get("top_k", 10),
                )
            )
        except BestPairSearchError as exc:
            return jsonify({"error": str(exc), "experiment": exc.payload}), exc.status_code
        except (FileNotFoundError, ValueError, TypeError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/datasets/<dataset_id>/reranking-ablation")
    def api_dataset_reranking_ablation(dataset_id: str):
        payload = request.get_json(silent=True) or {}
        try:
            job = start_reranking_ablation(
                dataset_id,
                payload.get("method_id", ""),
                payload.get("budgets") or [1, 4, 8, 16],
                scope=payload.get("scope", "train-test"),
                top_k=payload.get("top_k", 10),
            )
            return jsonify({"job": job}), 202
        except BestPairSearchError as exc:
            return jsonify({"error": str(exc), "experiment": exc.payload}), exc.status_code
        except (FileNotFoundError, ValueError, TypeError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/reranking-jobs/<job_id>")
    def api_reranking_job(job_id: str):
        try:
            return jsonify({"job": reranking_job(job_id)})
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404
        except (ValueError, json.JSONDecodeError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/datasets/<dataset_id>/evaluate")
    def api_dataset_evaluate(dataset_id: str):
        payload = request.get_json(silent=True) or {}
        method_ids = payload.get("methods") or [model["id"] for model in MODELS]
        sample_size = payload.get("sample_size", 12)
        scope = payload.get("scope", "train-test")
        try:
            return jsonify(
                evaluate_models(
                    dataset_id,
                    method_ids,
                    sample_size=sample_size,
                    scope=scope,
                    sample_mode=payload.get("sample_mode", "stratified"),
                    seed=payload.get("seed", 379),
                    top_k=payload.get("top_k", 5),
                )
            )
        except (FileNotFoundError, ValueError, TypeError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/benchmarks")
    def api_benchmarks():
        return jsonify({"benchmarks": benchmark_catalog()})

    @app.get("/api/benchmarks/<run_id>")
    def api_benchmark(run_id: str):
        try:
            return jsonify(load_benchmark(run_id))
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404
        except (ValueError, json.JSONDecodeError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/research-summary")
    def api_research_summary():
        return jsonify(
            {
                "summary": latest_research_summary(),
                "matrix_status": latest_research_matrix_status(),
                "checkpoint_audit": checkpoint_audit_summary(),
            }
        )

    @app.get("/api/training")
    def api_training():
        return jsonify(training_catalog(dataset_id=request.args.get("dataset")))

    @app.get("/api/hpo/catalog")
    def api_hpo_catalog():
        try:
            return jsonify(
                optimization_catalog(
                    request.args.get("dataset", ""),
                    request.args.get("model", ""),
                )
            )
        except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/training/start")
    def api_training_start():
        payload = request.get_json(silent=True) or {}
        try:
            return jsonify(
                start_training(
                    payload.get("model_id", ""),
                    payload.get("dataset_id", ""),
                    epochs=payload.get("epochs", 1),
                    batch_size=payload.get("batch_size", 32),
                    seed=payload.get("seed", 379),
                    optimize=bool(payload.get("optimize", False)),
                    trials=payload.get("trials", 6),
                    budget_mode=payload.get("budget_mode", "standard"),
                )
            )
        except (FileNotFoundError, ValueError, TypeError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/compare")
    def api_compare():
        payload = request.get_json(silent=True) or {}
        model_ids = payload.get("methods") or [model["id"] for model in MODELS]
        preview_only = bool(payload.get("preview_only", False))
        dataset_id = payload.get("dataset") or payload.get("dataset_id")
        meta = payload.get("meta") or {}

        try:
            left_payload, right_payload = _extract_graph_payloads(payload)
            left = graph_from_payload(left_payload, name="Graph A")
            right = graph_from_payload(right_payload, name="Graph B")
            results = (
                []
                if preview_only
                else run_models(left, right, model_ids, dataset_id=dataset_id, meta=meta)
            )
            input_matches_pair = original_pair_matches_graphs(
                dataset_id,
                meta.get("left_graph"),
                meta.get("right_graph"),
                left,
                right,
            )
            ground_truth = (
                pair_ground_truth(
                    dataset_id,
                    meta.get("left_graph"),
                    meta.get("right_graph"),
                    left.node_count,
                    right.node_count,
                )
                if input_matches_pair is not False
                else None
            )
        except (GraphInputError, ValueError, TypeError) as exc:
            return jsonify({"error": str(exc)}), 400

        return jsonify(
            {
                "results": results,
                "ground_truth": ground_truth,
                "input_matches_dataset_pair": input_matches_pair,
                "stats": {
                    "left": left.summary(),
                    "right": right.summary(),
                },
                "graphs": {
                    "left": left.to_preview(),
                    "right": right.to_preview(),
                },
            }
        )

    @app.post("/api/model-runs")
    def api_model_runs():
        payload = request.get_json(silent=True) or {}
        model_ids = payload.get("methods") or [model["id"] for model in MODELS]
        dataset_id = payload.get("dataset") or payload.get("dataset_id")
        try:
            left_payload, right_payload = _extract_graph_payloads(payload)
            job = start_model_run(
                dataset_id=dataset_id,
                model_ids=model_ids,
                left=graph_from_payload(left_payload, name="Graph A"),
                right=graph_from_payload(right_payload, name="Graph B"),
                meta=payload.get("meta") or {},
                hpo_mode=payload.get("hpo_mode") or "quick",
            )
            return jsonify({"job": job}), 202
        except (GraphInputError, FileNotFoundError, ValueError, TypeError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/model-runs/<job_id>")
    def api_model_run(job_id: str):
        try:
            return jsonify({"job": model_run_job(job_id)})
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return jsonify({"error": str(exc)}), 400

    return app


def _extract_graph_payloads(payload: dict):
    if "left" in payload and "right" in payload:
        return payload["left"], payload["right"]

    if isinstance(payload.get("graph_1"), list) and isinstance(payload.get("graph_2"), list):
        return (
            {"edges": payload["graph_1"], "labels": payload.get("labels_1", [])},
            {"edges": payload["graph_2"], "labels": payload.get("labels_2", [])},
        )

    raise GraphInputError("Request must include left/right graphs or graph_1/graph_2 edge lists.")


def _asset_version() -> str:
    mtimes = []
    for path in [
        BASE_DIR / "static" / "app.js",
        BASE_DIR / "static" / "styles.css",
        BASE_DIR / "static" / "itu-inspired.css",
    ]:
        try:
            mtimes.append(path.stat().st_mtime_ns)
        except OSError:
            continue
    return str(max(mtimes, default=0))
