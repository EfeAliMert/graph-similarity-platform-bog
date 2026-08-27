from collections import Counter
import json
import math
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from graph_similarity_platform import create_app
from graph_similarity_platform import evaluation
from graph_similarity_platform import research_summary
from graph_similarity_platform import search
from graph_similarity_platform.evaluation import _summarize_model
from scripts.audit_model_outputs import _audit_row
from scripts.universal_dataset import (
    build_pair_split,
    build_subject_disjoint_pair_split,
    canonicalize_symmetric_distances,
    canonical_pair_key,
    graph_disjoint_split_metadata,
    split_leakage_comparison,
)
from scripts.run_research_matrix import evaluation_seed_values
from graph_similarity_platform.research_summary import _reference_group
from scripts.prepare_simgnn_original_dataset import (
    balanced_sample,
    inject_identity_pairs,
    oversample_exact_zero_pairs,
    pair_key,
    validation_pool_with_training_reserve,
)


class ResearchProtocolTests(unittest.TestCase):
    def test_single_seed_standard_deviation_is_not_estimable(self):
        aggregate = research_summary._aggregate([1.25])

        self.assertIsNone(aggregate["std"])
        self.assertEqual(
            research_summary._mean_std(aggregate),
            "1.25 (std N/A; n=1)",
        )

    def test_ranking_metrics_include_all_ties_at_k_boundary(self):
        metrics = evaluation._ranking_metrics(
            exact_distances=[0.0, 0.0, 0.0, 2.0],
            predicted_distances=[0.1, 0.2, 0.0, 3.0],
            exact_relevance=[1.0, 1.0, 1.0, math.exp(-2.0)],
            k=2,
        )

        self.assertEqual(metrics["relevance_count"], 3)
        self.assertEqual(metrics["relevance_cutoff_distance"], 0.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 2.0 / 3.0)

    def test_retrieval_relevance_is_tie_aware(self):
        rows = [
            {"key": (0, 1), "exact_ged": 0.0},
            {"key": (0, 2), "exact_ged": 0.0},
            {"key": (1, 2), "exact_ged": 0.0},
            {"key": (2, 3), "exact_ged": 1.0},
        ]

        relevant, cutoff = search._tie_aware_relevant_keys(rows, top_k=1)

        self.assertEqual(cutoff, 0.0)
        self.assertEqual(relevant, {(0, 1), (0, 2), (1, 2)})

    def test_directional_ged_upper_bounds_are_canonicalized_symmetrically(self):
        distances = canonicalize_symmetric_distances(
            {(1, 2): 9.0, (2, 1): 6.0, (1, 1): 0.0}
        )

        self.assertEqual(distances[(1, 2)], 6.0)
        self.assertEqual(distances[(2, 1)], 6.0)
        self.assertEqual(distances[(1, 1)], 0.0)

    def test_model_output_audit_separates_integrity_from_accuracy(self):
        base = {
            "id": "simgnn",
            "name": "SimGNN",
            "status": "executed",
            "model_score": math.exp(-2.0),
            "canonical_similarity": math.exp(-2.0),
            "checkpoint_loaded": True,
            "architecture_loaded": True,
            "input_matches_dataset_pair": True,
            "score_semantics": "exp(-normalized GED)",
            "selected_checkpoint": "checkpoint.pt",
            "adapter_metrics": {"predicted_ged": 10.0, "seed": 379},
        }
        identity = {
            **base,
            "canonical_similarity": 0.4,
            "adapter_metrics": {"predicted_ged": 4.5, "seed": 379},
        }

        row = _audit_row(base, base, identity, {"distance": 2.0}, 5.0)

        self.assertTrue(row["technical_integrity"])
        self.assertEqual(row["exact_ged_error"], 8.0)
        self.assertEqual(row["identity_comparable_similarity"], 0.4)

    def test_paper_comparison_surface_and_endpoints_are_removed(self):
        client = create_app().test_client()

        home = client.get("/")
        self.assertEqual(home.status_code, 200)
        self.assertIn(b'id="loadingOverlay"', home.data)
        self.assertIn(b'id="loadingTitle"', home.data)
        self.assertNotIn(b"Paper vs Local", home.data)
        self.assertNotIn(b"Included Papers", home.data)
        app_js = (Path(__file__).parents[1] / "static" / "app.js").read_text()
        self.assertIn(
            "All active checkpoint protocols verified; accuracy is reported separately",
            app_js,
        )
        self.assertNotIn("All active checkpoints verified", app_js)
        self.assertEqual(client.get("/api/papers").status_code, 404)
        self.assertEqual(
            client.get("/api/paper-comparison/aids700nef").status_code,
            404,
        )

    def test_light_and_dark_interface_themes_are_available(self):
        client = create_app().test_client()
        home = client.get("/")
        self.assertEqual(home.status_code, 200)
        self.assertIn(b'id="themeSelect"', home.data)
        self.assertIn(b'<option value="light">Light</option>', home.data)
        self.assertIn(b'<option value="dark">Dark</option>', home.data)

        root = Path(__file__).parents[1]
        app_js = (root / "static" / "app.js").read_text()
        theme_css = (root / "static" / "itu-inspired.css").read_text()
        self.assertIn('localStorage.setItem(THEME_STORAGE_KEY, activeTheme)', app_js)
        self.assertIn('html[data-theme="dark"]', theme_css)

    def test_graph_disjoint_split_hash_is_deterministic_and_identity_sensitive(self):
        records = [
            {"id": graph_id, "split": "train"}
            for graph_id in range(8)
        ] + [{"id": 9, "split": "test"}]
        first = graph_disjoint_split_metadata(records, 0.25)
        second = graph_disjoint_split_metadata(records, 0.25)
        changed = graph_disjoint_split_metadata(
            [{**record, "id": 99} if record["id"] == 7 else record for record in records],
            0.25,
        )

        self.assertEqual(first["training_graphs"], 6)
        self.assertEqual(first["validation_graphs"], 2)
        self.assertEqual(first["pair_overlap"], 0)
        self.assertEqual(first["split_sha256"], second["split_sha256"])
        self.assertNotEqual(first["split_sha256"], changed["split_sha256"])

    def test_pair_holdout_is_reproducible_and_has_zero_symmetric_overlap(self):
        graphs = [
            {"id": graph_id, "nodes": list(range(graph_id + 2))}
            for graph_id in range(6)
        ]
        distances = {
            (left, right): float(abs(left - right) + 1)
            for left in range(6)
            for right in range(left + 1, 6)
        }

        first = build_pair_split(graphs, distances, validation_count=4, seed=379)
        second = build_pair_split(graphs, distances, validation_count=4, seed=379)

        self.assertEqual(first["validation_pairs"], second["validation_pairs"])
        self.assertEqual(first["metadata"]["split_sha256"], second["metadata"]["split_sha256"])
        self.assertEqual(first["metadata"]["pair_overlap"], 0)
        validation_keys = first["validation_keys"]
        training_keys = {
            canonical_pair_key(graphs[left]["id"], graphs[right]["id"])
            for left, right in first["training_pairs"]
        }
        self.assertTrue(validation_keys)
        self.assertTrue(training_keys)
        self.assertTrue(validation_keys.isdisjoint(training_keys))

    def test_pair_holdout_preserves_rare_exact_zero_stratum_for_training(self):
        graphs = [
            {"id": graph_id, "nodes": list(range(10))}
            for graph_id in range(50)
        ]
        candidate_keys = [
            (left, right)
            for left in range(50)
            for right in range(left + 1, 50)
        ][:123]
        distances = {
            key: (0.0 if index < 23 else 10.0)
            for index, key in enumerate(candidate_keys)
        }

        split = build_pair_split(
            graphs,
            distances,
            validation_count=40,
            seed=379,
        )
        zero_keys = set(candidate_keys[:23])
        validation_zero = zero_keys.intersection(split["validation_keys"])
        training_keys = {
            canonical_pair_key(graphs[left]["id"], graphs[right]["id"])
            for left, right in split["training_pairs"]
        }

        self.assertEqual(len(validation_zero), 5)
        self.assertEqual(len(zero_keys.intersection(training_keys)), 18)
        self.assertEqual(
            split["metadata"]["normalized_ged_bin_distribution"]["0"],
            {"candidates": 23, "training": 18, "validation": 5},
        )

    def test_subject_disjoint_split_has_zero_graph_and_pair_overlap(self):
        graphs = [
            {"id": graph_id, "nodes": list(range(19))}
            for graph_id in range(10)
        ]
        distances = {
            (left, right): float(abs(left - right) + 1)
            for left in range(10)
            for right in range(left + 1, 10)
        }

        first = build_subject_disjoint_pair_split(
            graphs,
            distances,
            validation_count=20,
            seed=379,
        )
        second = build_subject_disjoint_pair_split(
            graphs,
            distances,
            validation_count=20,
            seed=379,
        )

        metadata = first["metadata"]
        self.assertEqual(metadata["graph_overlap"], 0)
        self.assertEqual(metadata["pair_overlap"], 0)
        self.assertTrue(
            set(metadata["training_graph_ids"]).isdisjoint(
                metadata["validation_graph_ids"]
            )
        )
        self.assertEqual(metadata["split_sha256"], second["metadata"]["split_sha256"])

    def test_exact_ged_workflows_reject_registered_proxy_targets(self):
        with patch.object(search, "ground_truth_kind", return_value="structural_proxy"):
            with self.assertRaises(search.BestPairSearchError) as context:
                search.evaluate_prefilter_ablation("mutag", budgets=[1])
        self.assertEqual(context.exception.status_code, 422)
        self.assertFalse(context.exception.payload["ground_truth_exact"])

        with patch.object(evaluation, "ground_truth_kind", return_value="structural_proxy"):
            with self.assertRaisesRegex(ValueError, "structural proxy"):
                evaluation.evaluate_models("mutag", ["simgnn"])

    def test_simgnn_sampler_excludes_reverse_validation_pair(self):
        candidates = [
            (0, 1, 1.0, 0.5),
            (1, 0, 1.0, 0.5),
            (0, 2, 2.0, 0.8),
        ]
        sampled = balanced_sample(
            candidates,
            count=3,
            rng=random.Random(379),
            excluded={(0, 1)},
        )
        self.assertEqual([(left, right) for left, right, _, _ in sampled], [(0, 2)])

    def test_simgnn_split_keeps_and_oversamples_exact_zero_training_pairs(self):
        candidates = [
            (0, 1, 0.0, 0.0),
            (2, 3, 0.0, 0.0),
            (4, 5, 0.0, 0.0),
            (6, 7, 0.0, 0.0),
        ] + [
            (index, index + 20, float(index + 1), 0.5 + index * 0.1)
            for index in range(12)
        ]
        validation_candidates = validation_pool_with_training_reserve(
            candidates,
            random.Random(380),
        )
        validation = balanced_sample(
            validation_candidates,
            count=8,
            rng=random.Random(381),
        )
        validation_keys = {pair_key(left, right) for left, right, _, _ in validation}
        training = balanced_sample(
            candidates,
            count=12,
            rng=random.Random(379),
            excluded=validation_keys,
        )
        training = oversample_exact_zero_pairs(
            training,
            candidates,
            count=12,
            rng=random.Random(382),
            excluded=validation_keys,
            minimum_fraction=0.25,
        )

        self.assertEqual(len(training), 12)
        self.assertGreaterEqual(sum(pair[2] == 0 for pair in training), 3)
        self.assertTrue(
            validation_keys.isdisjoint(
                {pair_key(left, right) for left, right, _, _ in training}
            )
        )

        anchored = inject_identity_pairs(
            training,
            graph_ids=list(range(20)),
            count=12,
            rng=random.Random(383),
            fraction=0.25,
        )
        self.assertEqual(len(anchored), 12)
        self.assertEqual(sum(left == right for left, right, _, _ in anchored), 3)

    def test_benchmark_summary_reports_error_ranking_latency_and_ci(self):
        samples = []
        for index, (exact, predicted) in enumerate(((1.0, 1.2), (2.0, 2.1), (4.0, 3.8))):
            graph_size = 5.0
            exact_similarity = math.exp(-exact / graph_size)
            predicted_similarity = math.exp(-predicted / graph_size)
            samples.append(
                {
                    "status": "executed",
                    "average_graph_size": graph_size,
                    "exact_ged": exact,
                    "predicted_ged": predicted,
                    "exact_normalized_ged": exact / graph_size,
                    "predicted_normalized_ged": predicted / graph_size,
                    "exact_similarity": exact_similarity,
                    "predicted_similarity": predicted_similarity,
                    "abs_ged_error": abs(predicted - exact),
                    "abs_normalized_ged_error": abs(predicted - exact) / graph_size,
                    "abs_similarity_error": abs(predicted_similarity - exact_similarity),
                    "latency_ms": 10.0 + index,
                }
            )
        row = {"id": "test-model", "samples": samples}

        _summarize_model(row, top_k=2, seed=379)

        self.assertEqual(row["status"], "evaluated")
        self.assertAlmostEqual(row["spearman_ged"], 1.0)
        self.assertAlmostEqual(row["kendall_ged"], 1.0)
        self.assertAlmostEqual(row["precision_at_k"], 1.0)
        self.assertAlmostEqual(row["ndcg_at_k"], 1.0)
        self.assertEqual(row["latency_p50_ms"], 11.0)
        self.assertGreater(row["latency_p95_ms"], 11.0)
        self.assertEqual(len(row["mae_ged_ci95"]), 2)
        self.assertIsNotNone(row["mse_similarity_x1e3"])
        self.assertEqual(
            {bucket["bucket"] for bucket in row["size_generalization"]},
            {"small"},
        )
        self.assertGreater(row["throughput_pairs_per_second"], 0)

    def test_benchmark_artifact_catalog_and_api_are_local_and_loadable(self):
        with tempfile.TemporaryDirectory(
            dir=evaluation.BASE_DIR / "training_logs"
        ) as temporary, patch.object(
            evaluation,
            "BENCHMARK_DIR",
            Path(temporary),
        ):
            payload = {
                "run_id": "test-run-379",
                "dataset_id": "aids700nef",
                "completed_at": "2026-07-27T00:00:00+00:00",
                "sample_size": 3,
                "models": [{"id": "simgnn"}],
            }
            artifact_path = evaluation._persist_benchmark(payload)
            self.assertTrue((evaluation.BASE_DIR / artifact_path).exists())
            self.assertEqual(evaluation.benchmark_catalog()[0]["run_id"], "test-run-379")

            client = create_app().test_client()
            response = client.get("/api/benchmarks/test-run-379")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json()["dataset_id"], "aids700nef")

    def test_prefilter_ablation_reports_candidate_recall_and_regret(self):
        signature = {
            "nodes": 3,
            "edges": 2,
            "density": 0.5,
            "components": 1,
            "degrees": Counter({1: 2, 2: 1}),
            "labels": Counter({"C": 3}),
        }
        records = [
            {
                "id": 0,
                "split": "train",
                "member": "train/0.gexf",
                "signature": signature,
            },
            {
                "id": 1,
                "split": "train",
                "member": "train/1.gexf",
                "signature": {**signature, "nodes": 5},
            },
            {
                "id": 2,
                "split": "test",
                "member": "test/2.gexf",
                "signature": signature,
            },
        ]
        distances = {(0, 2): 1.0, (1, 2): 4.0}
        with tempfile.TemporaryDirectory(
            dir=search.BASE_DIR / "training_logs"
        ) as temporary, patch.object(
            search,
            "RETRIEVAL_DIR",
            Path(temporary),
        ), patch.object(
            search,
            "_load_records",
            return_value=records,
        ), patch.object(
            search,
            "load_ground_truth_distances",
            return_value=distances,
        ), patch.object(
            search,
            "ground_truth_kind",
            return_value="exact",
        ):
            payload = search.evaluate_prefilter_ablation(
                "test",
                budgets=[1, 2],
                top_k=1,
            )

        self.assertEqual(payload["total_pairs"], 2)
        self.assertEqual(payload["budgets"][0]["recall_at_k"], 1.0)
        self.assertTrue(payload["budgets"][0]["exact_best_recalled"])
        self.assertEqual(payload["budgets"][0]["best_ged_regret"], 0.0)

    def test_real_gnn_reranking_reports_candidate_and_model_metrics(self):
        signature = {
            "nodes": 3,
            "edges": 2,
            "density": 0.5,
            "components": 1,
            "degrees": Counter({1: 2, 2: 1}),
            "labels": Counter({"C": 3}),
        }
        records = [
            {"id": 0, "split": "train", "member": "train/0.gexf", "signature": signature, "data": object()},
            {"id": 1, "split": "train", "member": "train/1.gexf", "signature": {**signature, "nodes": 5}, "data": object()},
            {"id": 2, "split": "test", "member": "test/2.gexf", "signature": signature, "data": object()},
        ]
        distances = {(0, 2): 1.0, (1, 2): 4.0}
        scores = iter((0.9, 0.2))
        with tempfile.TemporaryDirectory(
            dir=search.BASE_DIR / "training_logs"
        ) as temporary, patch.object(
            search,
            "RETRIEVAL_DIR",
            Path(temporary),
        ), patch.object(
            search,
            "_load_records",
            return_value=records,
        ), patch.object(
            search,
            "load_ground_truth_distances",
            return_value=distances,
        ), patch.object(
            search,
            "ground_truth_kind",
            return_value="exact",
        ), patch.object(
            search,
            "_runnable_methods",
            return_value=["simgnn"],
        ), patch.object(
            search,
            "load_original_pair",
            return_value={"meta": {}},
        ), patch.object(
            search,
            "run_models",
            side_effect=lambda *_args, **_kwargs: [
                {
                    "status": "executed",
                    "canonical_similarity": next(scores),
                    "latency_ms": 2.0,
                    "adapter_metrics": {"predicted_ged": 1.0},
                    "selected_checkpoint": "checkpoint.pt",
                }
            ],
        ):
            payload = search.evaluate_reranking_ablation(
                "test",
                "simgnn",
                budgets=[1, 2],
                top_k=1,
            )

        self.assertEqual(payload["method_id"], "simgnn")
        self.assertEqual(payload["budgets"][0]["candidate_recall_at_k"], 1.0)
        self.assertEqual(payload["budgets"][0]["reranked_recall_at_k"], 1.0)
        self.assertEqual(payload["budgets"][0]["model_selected_ged_regret"], 0.0)

    def test_all_model_reranking_ensemble_requires_and_averages_every_member(self):
        signature = {
            "nodes": 3,
            "edges": 2,
            "density": 0.5,
            "components": 1,
            "degrees": Counter({1: 2, 2: 1}),
            "labels": Counter({"C": 3}),
        }
        records = [
            {"id": 0, "split": "train", "member": "train/0.gexf", "signature": signature, "data": object()},
            {"id": 1, "split": "test", "member": "test/1.gexf", "signature": signature, "data": object()},
        ]
        method_ids = ["simgnn", "multiscale-set", "segmn", "graph-fusion", "graph2region"]
        results = [
            {
                "id": method_id,
                "status": "executed",
                "canonical_similarity": score,
                "latency_ms": 1.0,
                "selected_checkpoint": f"{method_id}.pt",
            }
            for method_id, score in zip(method_ids, (0.6, 0.7, 0.8, 0.9, 1.0))
        ]
        with tempfile.TemporaryDirectory(
            dir=search.BASE_DIR / "training_logs"
        ) as temporary, patch.object(
            search,
            "RETRIEVAL_DIR",
            Path(temporary),
        ), patch.object(
            search,
            "_load_records",
            return_value=records,
        ), patch.object(
            search,
            "load_ground_truth_distances",
            return_value={(0, 1): 1.0},
        ), patch.object(
            search,
            "ground_truth_kind",
            return_value="exact",
        ), patch.object(
            search,
            "_runnable_methods",
            return_value=method_ids,
        ), patch.object(
            search,
            "load_original_pair",
            return_value={"meta": {}},
        ), patch.object(
            search,
            "run_models",
            return_value=results,
        ):
            payload = search.evaluate_reranking_ablation(
                "test",
                search.ENSEMBLE_METHOD_ID,
                budgets=[1],
                top_k=1,
            )

        self.assertEqual(payload["method_ids"], method_ids)
        self.assertEqual(payload["budgets"][0]["model_selected_pair"]["model_score"], 0.8)
        self.assertEqual(payload["budgets"][0]["latency_total_ms"], 5.0)

    def test_reranker_rejects_graphsim_score_when_ged_is_undefined(self):
        result = {
            "id": "multiscale-set",
            "status": "executed",
            "score": 1.0,
            "canonical_similarity": None,
            "adapter_metrics": {"raw_score": 1.2},
        }

        member = search._reranker_result_score(result)

        self.assertIsNone(member["score"])
        self.assertEqual(member["source"], "unavailable")

    def test_latest_research_summary_prefers_completed_evaluated_matrix(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            research_summary,
            "SUMMARY_DIR",
            Path(temporary),
        ):
            completed = {
                "run_id": "completed-exact",
                "matrix_complete": True,
                "rows": [{"evaluated_seeds": [379, 2026, 3407]}],
            }
            newer_training_only = {
                "run_id": "newer-training-only",
                "matrix_complete": False,
                "rows": [{"evaluated_seeds": []}],
            }
            completed_path = Path(temporary) / "completed.json"
            training_path = Path(temporary) / "training.json"
            completed_path.write_text(json.dumps(completed))
            training_path.write_text(json.dumps(newer_training_only))
            training_path.touch()

            selected = research_summary.latest_research_summary()

        self.assertEqual(selected["run_id"], "completed-exact")

    def test_matrix_status_matches_selected_completed_summary(self):
        completed = {
            "run_id": "exact-complete",
            "execute": True,
            "datasets": ["aids700nef"],
            "models": ["simgnn"],
            "seeds": [379],
            "jobs": [{"status": "completed"}],
        }
        newer_proxy = {
            "run_id": "proxy-complete",
            "execute": True,
            "datasets": ["mutag"],
            "models": ["simgnn"],
            "seeds": [379],
            "jobs": [{"status": "completed"}],
        }
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            research_summary,
            "MATRIX_DIR",
            Path(temporary),
        ), patch.object(
            research_summary,
            "latest_research_summary",
            return_value={"run_id": "exact-complete"},
        ):
            exact_path = Path(temporary) / "exact" / "manifest.json"
            proxy_path = Path(temporary) / "proxy" / "manifest.json"
            exact_path.parent.mkdir()
            proxy_path.parent.mkdir()
            exact_path.write_text(json.dumps(completed))
            proxy_path.write_text(json.dumps(newer_proxy))
            proxy_path.touch()

            status = research_summary.latest_research_matrix_status()

        self.assertEqual(status["run_id"], "exact-complete")

    def test_evaluate_existing_uses_a_single_sampling_seed(self):
        self.assertEqual(
            evaluation_seed_values(True, [379, 2026, 3407], 379),
            [379],
        )
        self.assertEqual(
            evaluation_seed_values(False, [379, 2026], 379),
            [379, 2026],
        )

    def test_research_summary_keeps_exact_and_approximate_tables_apart(self):
        self.assertEqual(
            _reference_group({"reference_kind": "exact", "target_source": "exact A* GED"}),
            "exact",
        )
        self.assertEqual(
            _reference_group(
                {
                    "reference_kind": "approximate_benchmark",
                    "target_source": "approximate GED benchmark upper bound",
                }
            ),
            "approximate",
        )

    def test_pair_disjoint_validation_can_share_graphs_but_subject_disjoint_does_not(self):
        graphs = [
            {"id": graph_id, "nodes": list(range(8)), "subject_id": graph_id // 2}
            for graph_id in range(12)
        ]
        distances = {
            (left, right): float(abs(left - right) + 1)
            for left in range(12)
            for right in range(left + 1, 12)
        }
        comparison = split_leakage_comparison(
            graphs,
            distances,
            validation_count=16,
            seed=379,
        )

        self.assertGreater(comparison["pair_disjoint"]["graph_overlap"], 0)
        self.assertEqual(comparison["pair_disjoint"]["pair_overlap"], 0)
        self.assertEqual(comparison["subject_disjoint"]["graph_overlap"], 0)
        self.assertEqual(comparison["subject_disjoint"]["pair_overlap"], 0)

    def test_summarize_model_separates_projected_and_unprojected_pairs(self):
        row = {
            "id": "segmn",
            "samples": [
                _evaluation_sample(ged=1.0, predicted=1.2, projected=False),
                _evaluation_sample(ged=2.0, predicted=2.5, projected=False),
                _evaluation_sample(ged=8.0, predicted=20.0, projected=True),
            ],
        }
        _summarize_model(row, top_k=2, seed=379)
        self.assertEqual(row["projected_samples"], 1)
        self.assertEqual(row["unprojected_metrics"]["samples"], 2)
        self.assertEqual(row["projected_metrics"]["samples"], 1)
        self.assertLess(
            row["unprojected_metrics"]["mae_ged"],
            row["projected_metrics"]["mae_ged"],
        )


def _evaluation_sample(ged: float, predicted: float, projected: bool) -> dict:
    graph_size = 10.0
    return {
        "status": "executed",
        "exact_ged": ged,
        "predicted_ged": predicted,
        "exact_normalized_ged": ged / graph_size,
        "predicted_normalized_ged": predicted / graph_size,
        "exact_similarity": math.exp(-ged / graph_size),
        "abs_ged_error": abs(predicted - ged),
        "abs_normalized_ged_error": abs((predicted - ged) / graph_size),
        "abs_similarity_error": abs(
            math.exp(-predicted / graph_size) - math.exp(-ged / graph_size)
        ),
        "projection_applied": projected,
        "average_graph_size": graph_size,
        "latency_ms": 1.0,
        "pair_split": {"pair_overlap": 0},
        "checkpoint_seed": 379,
    }


if __name__ == "__main__":
    unittest.main()
