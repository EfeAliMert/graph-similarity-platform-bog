import copy
import math
import os
import subprocess
import unittest
from unittest.mock import patch

from graph_similarity_platform import create_app
from graph_similarity_platform.evaluation import _canonical_similarity
from scripts.graphsim_calibration import (
    CALIBRATION_METHOD,
    apply_isotonic_calibration,
    calibration_position,
    validate_calibration,
)
from graph_similarity_platform.graph_utils import graph_from_payload
from graph_similarity_platform.models.real_models import (
    BASE_DIR,
    MODELS,
    _canonical_similarity as _runtime_canonical_similarity,
    _execute_json_adapter,
    _preferred_checkpoint,
    _resource_metrics,
    _target_note,
    run_models,
)
from graph_similarity_platform.search import _aggregate_score
from graph_similarity_platform.evaluation import _summarize_model
from graph_similarity_platform.training import _pid_is_running, training_catalog
from scripts.audit_checkpoints import (
    _graphsim_calibration_verified,
    _portable_metadata,
)


def _require_local_artifacts(testcase, condition, message):
    if condition:
        return
    if os.environ.get("GSP_REQUIRE_ARTIFACTS") == "1":
        testcase.fail(message)
    testcase.skipTest(message)


class RealModelRegistryTests(unittest.TestCase):
    def test_checkpoint_audit_converts_local_absolute_paths(self):
        value = _portable_metadata(
            {"checkpoint": str(BASE_DIR / "training_logs" / "trial" / "model.pt")}
        )

        self.assertEqual(
            value["checkpoint"],
            "training_logs/trial/model.pt",
        )

    def test_preview_only_compare_does_not_execute_models(self):
        client = create_app().test_client()
        with patch("graph_similarity_platform.run_models") as execute_models:
            response = client.post(
                "/api/compare",
                json={
                    "left": {"edges": [[0, 1]]},
                    "right": {"edges": [[0, 1], [1, 2]]},
                    "preview_only": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        execute_models.assert_not_called()
        payload = response.get_json()
        self.assertEqual(payload["results"], [])
        self.assertEqual(payload["stats"]["left"]["nodes"], 2)
        self.assertEqual(payload["stats"]["right"]["nodes"], 3)

    def test_graphsim_validation_calibration_is_monotonic_and_bounded(self):
        calibration = {
            "method": CALIBRATION_METHOD,
            "x_thresholds": [0.0, 1.0, 2.0],
            "y_thresholds": [0.1, 0.4, 0.8],
        }
        self.assertAlmostEqual(
            apply_isotonic_calibration(1.5, calibration),
            0.6,
        )
        self.assertEqual(apply_isotonic_calibration(-5.0, calibration), 0.1)
        self.assertEqual(apply_isotonic_calibration(5.0, calibration), 0.8)
        self.assertEqual(calibration_position(1.5, calibration), "within_fit_range")
        self.assertEqual(calibration_position(3.0, calibration), "above_fit_range")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            apply_isotonic_calibration(float("nan"), calibration)

    def test_graphsim_requires_a_validation_calibrator(self):
        with self.assertRaisesRegex(ValueError, "no validation calibration"):
            validate_calibration(None)

    def test_checkpoint_audit_requires_leakage_free_graphsim_calibration(self):
        calibration = {
            "method": CALIBRATION_METHOD,
            "x_thresholds": [0.0, 1.0],
            "y_thresholds": [0.2, 0.8],
            "fit_audit_graph_overlap": 0,
            "fit_graph_ids": [1, 2],
            "audit_graph_ids": [3],
            "test_graphs_used": False,
            "fit_pair_count": 256,
            "audit_pair_count": 128,
            "fit_mse_calibrated": 0.02,
            "audit_mse_raw": 0.04,
            "audit_mse_calibrated": 0.03,
        }
        self.assertTrue(_graphsim_calibration_verified(calibration))
        self.assertFalse(
            _graphsim_calibration_verified(
                {**calibration, "test_graphs_used": True}
            )
        )
        self.assertFalse(
            _graphsim_calibration_verified(
                {**calibration, "audit_mse_calibrated": 0.05}
            )
        )
        self.assertTrue(
            _graphsim_calibration_verified(
                {
                    **calibration,
                    "accepted_by_audit": False,
                    "audit_mse_calibrated": 0.05,
                }
            )
        )

    def test_training_pid_probe_distinguishes_live_and_missing_processes(self):
        with patch("graph_similarity_platform.training.os.kill") as kill:
            self.assertTrue(_pid_is_running(123))
            kill.assert_called_once_with(123, 0)
        with patch(
            "graph_similarity_platform.training.os.kill",
            side_effect=ProcessLookupError,
        ):
            self.assertFalse(_pid_is_running(123))
        with patch(
            "graph_similarity_platform.training.os.kill",
            side_effect=PermissionError,
        ):
            self.assertTrue(_pid_is_running(123))
        self.assertFalse(_pid_is_running(None))

    def test_resource_metrics_parse_macos_and_linux_time_output(self):
        self.assertEqual(
            _resource_metrics("  123456789  maximum resident set size\n"),
            {"peak_rss_bytes": 123456789},
        )
        self.assertEqual(
            _resource_metrics("Maximum resident set size (kbytes): 2048\n"),
            {"peak_rss_bytes": 2 * 1024 * 1024},
        )

    def test_target_note_accepts_both_zero_overlap_metadata_keys(self):
        note = _target_note(
            {
                "target": {"target_source": "exact GED", "exact": True},
                "pair_split": {"pair_overlap_count": 0},
            }
        )
        self.assertIn("verified zero pair overlap", note)

    def test_adapter_timeout_becomes_model_failure(self):
        with patch(
            "graph_similarity_platform.models.real_models.subprocess.run",
            side_effect=subprocess.TimeoutExpired(["model"], 12),
        ):
            metrics, error = _execute_json_adapter(
                ["model"],
                cwd=BASE_DIR,
                timeout=12,
                label="Test Model",
            )
        self.assertIsNone(metrics)
        self.assertEqual(error["status"], "adapter_failed")
        self.assertIn("timed out after 12 seconds", error["detail"])

    def test_best_pair_ignores_executed_results_without_canonical_score(self):
        results = [
            {
                "status": "executed",
                "score": 1.0,
                "canonical_similarity": None,
            },
            {
                "status": "executed",
                "score": 0.4,
                "canonical_similarity": 0.4,
            },
        ]
        self.assertEqual(_aggregate_score(results), 0.4)

    def test_executed_invalid_regression_is_not_evaluable(self):
        row = {
            "samples": [
                {
                    "status": "executed",
                    "model_score": 1.0,
                    "predicted_similarity": None,
                    "predicted_ged": None,
                }
            ]
        }
        _summarize_model(row)
        self.assertEqual(row["status"], "not_evaluable")
        self.assertEqual(row["executed_samples"], 1)
        self.assertEqual(row["evaluated_samples"], 0)

    def test_every_registered_dataset_has_five_trainable_checkpoints(self):
        dataset_ids = {
            "aids700nef",
            "linux",
            "imdbmulti",
            "ptc",
            "mutag",
            "proteins",
            "enzymes",
        }
        missing_sources = [
            model["id"]
            for model in MODELS
            if not (BASE_DIR / model["local_path"] / model["entrypoint"]).exists()
        ]
        _require_local_artifacts(
            self,
            not missing_sources,
            "Local model source bundle is not installed: " + ", ".join(missing_sources),
        )

        missing = []
        for dataset_id in dataset_ids:
            plans = training_catalog(dataset_id)["plans"]
            self.assertEqual(len(plans), 5)
            self.assertTrue(all(plan["can_start"] for plan in plans))
            for model in MODELS:
                self.assertIn(dataset_id, model["datasets"])
                checkpoint = _preferred_checkpoint(
                    BASE_DIR / model["local_path"],
                    model,
                    dataset_id,
                )
                if checkpoint is None:
                    missing.append(f"{model['id']}:{dataset_id}")

        _require_local_artifacts(
            self,
            not missing,
            "Local checkpoint bundle is not installed: " + ", ".join(missing),
        )

    def test_only_real_models_are_listed(self):
        missing = []
        for model in MODELS:
            local_path = BASE_DIR / model["local_path"]
            checkpoint = _preferred_checkpoint(
                local_path,
                model,
                "aids700nef",
            )
            if not local_path.exists() or checkpoint is None:
                missing.append(model["id"])
        _require_local_artifacts(
            self,
            not missing,
            "AIDS700nef model artifacts are not installed: " + ", ".join(missing),
        )

        graph = {"edges": [[0, 1], [1, 2], [2, 0]], "labels": ["A", "B", "A"]}
        left = graph_from_payload(graph, name="left")
        right = graph_from_payload(graph, name="right")

        results = run_models(left, right, [model["id"] for model in MODELS], dataset_id="aids700nef")

        self.assertEqual(len(results), 5)
        self.assertTrue(all(not result["missing_runtime"] for result in results))
        self.assertEqual(
            {result["id"] for result in results},
            {"simgnn", "multiscale-set", "segmn", "graph-fusion", "graph2region"},
        )
        simgnn = next(result for result in results if result["id"] == "simgnn")
        self.assertIsInstance(simgnn["score"], float)
        self.assertEqual(simgnn["status"], "executed")
        self.assertTrue(simgnn["architecture_loaded"])
        self.assertTrue(simgnn["checkpoint_loaded"])
        self.assertFalse(simgnn["official_pretrained"])
        self.assertEqual(simgnn["checkpoint_origin"], "Locally trained in this workspace")
        self.assertIn("SimGNN", simgnn["architecture_class"])

    def test_accuracy_uses_canonical_similarity_not_raw_kernel_score(self):
        graph_size = 5.0
        result = {
            "canonical_similarity": None,
            "adapter_metrics": {"predicted_ged": 10.0},
        }
        self.assertAlmostEqual(
            _canonical_similarity(result, predicted_ged=10.0, graph_size=graph_size),
            math.exp(-2.0),
        )

    def test_runtime_canonical_similarity_rejects_invalid_ged(self):
        base = {"status": "executed", "adapter_metrics": {}}
        self.assertIsNone(
            _runtime_canonical_similarity(
                {**base, "adapter_metrics": {"predicted_ged": -1.0}},
                5.0,
            )
        )
        self.assertIsNone(
            _runtime_canonical_similarity(
                {**base, "adapter_metrics": {"predicted_ged": float("nan")}},
                5.0,
            )
        )
        self.assertAlmostEqual(
            _runtime_canonical_similarity(
                {**base, "adapter_metrics": {"predicted_ged": 10.0}},
                5.0,
            ),
            math.exp(-2.0),
        )

    def test_model_statuses_are_reported(self):
        left = graph_from_payload({"edges": [[0, 1], [1, 2]], "labels": ["A", "A", "A"]})
        right = graph_from_payload({"edges": [[0, 1], [1, 2], [2, 3], [3, 4]], "labels": ["B"] * 5})

        results = run_models(left, right, ["simgnn", "segmn", "graph-fusion"], dataset_id="ptc")

        self.assertEqual(len(results), 3)
        self.assertTrue(all("status" in result for result in results))
        self.assertTrue(all("detail" in result for result in results))
        self.assertIn("dataset_supported", results[0])

    def test_original_dataset_surface(self):
        original_dir = (
            BASE_DIR
            / "Models&Datasets"
            / "drive-download-20260630T100606Z-3-001"
        )
        required_dataset_files = [
            original_dir / "AIDS700nef.zip",
            original_dir / "LINUX.tar.gz",
            original_dir / "IMDBMulti.zip",
            original_dir / "PTC.zip",
            original_dir / "MUTAG.zip",
            original_dir / "PROTEINS.zip",
            original_dir / "ENZYMES.zip",
            original_dir / "aids700nef_ged_astar_gidpair_dist_map.pickle",
        ]
        simgnn = next(model for model in MODELS if model["id"] == "simgnn")
        simgnn_path = BASE_DIR / simgnn["local_path"]
        has_artifacts = (
            all(path.exists() for path in required_dataset_files)
            and simgnn_path.exists()
            and _preferred_checkpoint(simgnn_path, simgnn, "aids700nef") is not None
        )
        _require_local_artifacts(
            self,
            has_artifacts,
            "Registered datasets and the AIDS700nef SimGNN artifacts are required.",
        )

        app = create_app()
        client = app.test_client()

        datasets = client.get("/api/datasets").get_json()["datasets"]
        self.assertEqual(
            {dataset["id"] for dataset in datasets},
            {
                "aids700nef",
                "linux",
                "imdbmulti",
                "ptc",
                "mutag",
                "proteins",
                "enzymes",
            },
        )
        self.assertTrue(all(dataset["graph_count"] > 0 for dataset in datasets))
        self.assertTrue(any(dataset["id"] == "mutag" and dataset["graph_count"] == 188 for dataset in datasets))

        mutag_response = client.get("/api/datasets/mutag")
        mutag_payload = mutag_response.get_json()
        self.assertEqual(mutag_response.status_code, 200)
        mutag_compare = client.post(
            "/api/compare",
            json={
                "dataset": "mutag",
                "left": mutag_payload["left"],
                "right": mutag_payload["right"],
                "meta": mutag_payload["meta"],
                "methods": ["simgnn", "graph-fusion", "graph2region"],
            },
        )
        mutag_compare_payload = mutag_compare.get_json()
        self.assertEqual(mutag_compare.status_code, 200)
        self.assertIsNone(mutag_compare_payload["ground_truth"])
        self.assertGreater(mutag_compare_payload["stats"]["left"]["nodes"], 0)

        mutag_exact_best = client.post(
            "/api/datasets/mutag/best-pair",
            json={"methods": ["exact-ged"], "max_pairs": 3, "scope": "train-test"},
        )
        mutag_exact_payload = mutag_exact_best.get_json()
        self.assertEqual(mutag_exact_best.status_code, 422)
        self.assertIn("registered GED benchmark", mutag_exact_payload["error"])
        self.assertFalse(mutag_exact_payload["search"]["ground_truth_available"])

        mutag_structure_best = client.post(
            "/api/datasets/mutag/best-pair",
            json={"methods": ["structure-search"], "max_pairs": 3, "scope": "train-test"},
        )
        mutag_structure_payload = mutag_structure_best.get_json()
        self.assertEqual(mutag_structure_best.status_code, 200)
        self.assertEqual(mutag_structure_payload["search"]["method_ids"], ["structure-search"])
        self.assertEqual(mutag_structure_payload["search"]["scored_pairs"], mutag_structure_payload["search"]["total_pairs"])
        self.assertEqual(mutag_structure_payload["search"]["winner"]["results"][0]["status"], "structural_search")
        self.assertGreater(mutag_structure_payload["search"]["winner"]["score"], 0)

        mutag_evaluation = client.post(
            "/api/datasets/mutag/evaluate",
            json={"methods": ["simgnn"], "sample_size": 1, "scope": "train-test"},
        )
        mutag_evaluation_payload = mutag_evaluation.get_json()
        self.assertEqual(mutag_evaluation.status_code, 400)
        self.assertIn("structural proxy", mutag_evaluation_payload["error"])

        response = client.get("/api/datasets/aids700nef")
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["meta"]["dataset"], "AIDS700nef")
        self.assertIn("AIDS700nef/", payload["meta"]["left_graph"])

        graphs_response = client.get("/api/datasets/aids700nef/graphs")
        graphs_payload = graphs_response.get_json()
        self.assertEqual(graphs_response.status_code, 200)
        self.assertGreater(len(graphs_payload["train"]), 1)
        self.assertGreater(len(graphs_payload["test"]), 1)

        chosen_left = graphs_payload["train"][1]["member"]
        chosen_right = graphs_payload["test"][1]["member"]
        pair_response = client.get(
            "/api/datasets/aids700nef/pair",
            query_string={"left": chosen_left, "right": chosen_right},
        )
        pair_payload = pair_response.get_json()
        self.assertEqual(pair_response.status_code, 200)
        self.assertEqual(pair_payload["meta"]["left_graph"], chosen_left)
        self.assertEqual(pair_payload["meta"]["right_graph"], chosen_right)

        best_response = client.post(
            "/api/datasets/aids700nef/best-pair",
            json={"methods": ["simgnn"], "max_pairs": 1, "scope": "train-test"},
        )
        best_payload = best_response.get_json()
        self.assertEqual(best_response.status_code, 200)
        self.assertEqual(best_payload["meta"]["dataset_id"], "aids700nef")
        self.assertTrue(best_payload["meta"]["best_pair"])
        self.assertEqual(best_payload["search"]["scored_pairs"], 1)
        self.assertFalse(best_payload["search"]["exhaustive"])
        self.assertIsInstance(best_payload["search"]["winner"]["score"], float)

        exact_best = client.post(
            "/api/datasets/aids700nef/best-pair",
            json={"methods": ["exact-ged"], "max_pairs": 3, "scope": "train-test"},
        )
        exact_payload = exact_best.get_json()
        self.assertEqual(exact_best.status_code, 200)
        self.assertEqual(exact_payload["search"]["method_ids"], ["exact-ged"])
        self.assertEqual(exact_payload["search"]["scored_pairs"], exact_payload["search"]["total_pairs"])
        self.assertTrue(exact_payload["search"]["exhaustive"])
        self.assertLessEqual(exact_payload["search"]["displayed_pairs"], 3)
        self.assertIn("exact_ged", exact_payload["search"]["winner"])
        self.assertIn("left", exact_payload)
        self.assertIn("right", exact_payload)

        evaluation = client.post(
            "/api/datasets/aids700nef/evaluate",
            json={"methods": ["simgnn"], "sample_size": 1, "scope": "train-test"},
        )
        evaluation_payload = evaluation.get_json()
        self.assertEqual(evaluation.status_code, 200)
        self.assertEqual(evaluation_payload["sample_size"], 1)
        self.assertEqual(evaluation_payload["models"][0]["id"], "simgnn")
        self.assertGreaterEqual(evaluation_payload["models"][0]["executed_samples"], 1)
        self.assertIn("mae_similarity", evaluation_payload["models"][0])

        training = client.get("/api/training", query_string={"dataset": "aids700nef"})
        training_payload = training.get_json()
        self.assertEqual(training.status_code, 200)
        self.assertTrue(any(plan["id"] == "simgnn" and plan["can_start"] for plan in training_payload["plans"]))
        self.assertTrue(
            any(plan["id"] == "multiscale-set" and plan["can_start"] for plan in training_payload["plans"])
        )

        compare = client.post(
            "/api/compare",
            json={
                "dataset": "aids700nef",
                "left": payload["left"],
                "right": payload["right"],
                "meta": payload["meta"],
                "methods": ["simgnn", "multiscale-set", "segmn", "graph-fusion", "graph2region"],
            },
        )
        compare_payload = compare.get_json()
        self.assertEqual(compare.status_code, 200)
        self.assertIsNotNone(compare_payload["ground_truth"])
        self.assertIn("distance", compare_payload["ground_truth"])
        self.assertIn("similarity", compare_payload["ground_truth"])
        self.assertEqual(
            {result["id"]: result["status"] for result in compare_payload["results"]},
            {
                "simgnn": "executed",
                "multiscale-set": "executed",
                "segmn": "executed",
                "graph-fusion": "executed",
                "graph2region": "executed",
            },
        )
        self.assertTrue(compare_payload["input_matches_dataset_pair"])
        self.assertTrue(all(not result["official_pretrained"] for result in compare_payload["results"]))
        self.assertTrue(all(result["architecture_loaded"] for result in compare_payload["results"]))
        self.assertTrue(all(result["checkpoint_loaded"] for result in compare_payload["results"]))
        self.assertTrue(
            all(result["selected_checkpoint"] for result in compare_payload["results"])
        )
        self.assertTrue(
            all(
                isinstance(result["canonical_similarity"], float)
                for result in compare_payload["results"]
                if result["adapter_metrics"].get("predicted_ged") is not None
            )
        )
        self.assertTrue(
            all(
                result["score_transformation"]["native_score"]
                == result["model_score"]
                for result in compare_payload["results"]
            )
        )
        self.assertTrue(
            all(
                result["score_transformation"]["canonical_similarity"]
                == result["canonical_similarity"]
                for result in compare_payload["results"]
            )
        )
        graphsim = next(
            result
            for result in compare_payload["results"]
            if result["id"] == "multiscale-set"
        )
        calibration_applied = bool(
            graphsim["adapter_metrics"]["calibration_applied"]
        )
        self.assertEqual(
            graphsim["adapter_metrics"]["calibration_rejected_by_audit"],
            not calibration_applied,
        )
        self.assertEqual(
            graphsim["score_transformation"]["raw_model_output"],
            graphsim["adapter_metrics"]["raw_score"],
        )
        self.assertEqual(
            graphsim["score_transformation"]["calibration_applied"],
            calibration_applied,
        )
        self.assertEqual(
            graphsim["adapter_metrics"]["ged_prediction_source"],
            (
                "validation_isotonic_calibration"
                if calibration_applied
                else "native_output_calibration_rejected_by_audit"
            ),
        )
        self.assertFalse(
            graphsim["adapter_metrics"]["calibration_test_graphs_used"]
        )

        edited_left = copy.deepcopy(payload["left"])
        edited_left["edges"] = edited_left["edges"][1:]
        edited_compare = client.post(
            "/api/compare",
            json={
                "dataset": "aids700nef",
                "left": edited_left,
                "right": payload["right"],
                "meta": payload["meta"],
                "methods": ["simgnn", "multiscale-set"],
            },
        )
        edited_payload = edited_compare.get_json()
        self.assertEqual(edited_compare.status_code, 200)
        self.assertFalse(edited_payload["input_matches_dataset_pair"])
        self.assertIsNone(edited_payload["ground_truth"])
        self.assertEqual(
            {result["id"]: result["status"] for result in edited_payload["results"]},
            {"simgnn": "executed", "multiscale-set": "input_mismatch"},
        )

        self.assertEqual(client.get("/api/samples").status_code, 404)


if __name__ == "__main__":
    unittest.main()
