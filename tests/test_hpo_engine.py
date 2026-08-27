from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from graph_similarity_platform.hpo.adapters.base import (
    ModelHPOAdapter,
    ParameterCapability,
    batch_choices,
)
from graph_similarity_platform.hpo.registry import SearchSpaceRegistry
from graph_similarity_platform.hpo.best_config import BestConfigRegistry
from graph_similarity_platform.hpo.fingerprint import hash_file
from graph_similarity_platform.hpo.optimizer import (
    HyperparameterOptimizer,
    OptimizationRequest,
)
from graph_similarity_platform.hpo.reproducibility import seed_everything
from graph_similarity_platform.hpo.storage import ProgressStore, stable_study_name
from graph_similarity_platform.hpo.trial_runner import _parse_intermediate
from graph_similarity_platform.hpo.types import (
    DatasetProfile,
    DistributionStats,
    TrialContext,
    TrialResult,
)
from scripts.universal_dataset import (
    build_pair_split,
    canonical_pair_key,
)


def profile(fingerprint: str = "a" * 64) -> DatasetProfile:
    distribution = DistributionStats(3, 1.0, 1.5, 2.0, 2.0, 2.5, 3.0, 0.8)
    return DatasetProfile(
        dataset_id="unit-dataset",
        dataset_name="Unit Dataset",
        fingerprint=fingerprint,
        profile_version="dataset-profile-v1",
        target_kind="exact",
        target_source="unit-test exact GED",
        target_exact=True,
        split_strategy="canonical unordered pair holdout",
        graph_count=8,
        train_graph_count=6,
        test_graph_count=2,
        node_count=distribution,
        edge_count=distribution,
        density=distribution,
        degree=distribution,
        connected_components=distribution,
        node_label_cardinality=2,
        edge_label_cardinality=0,
        node_features_available=False,
        edge_labels_available=False,
        ged=distribution,
        normalized_ged=distribution,
        target_variance=0.25,
        zero_target_fraction=0.1,
        preprocessing={},
    )


class FakeProfiler:
    def __init__(self, value: DatasetProfile):
        self.value = value

    def profile(self, *_args, **_kwargs) -> DatasetProfile:
        return self.value


class FakeAdapter(ModelHPOAdapter):
    model_id = "graph-fusion"
    display_name = "Fake Graph Fusion"
    search_space_version = "fake-v1"

    def suggest(self, trial, _profile):
        return {"learning_rate": trial.suggest_categorical("learning_rate", [0.1, 0.2])}

    def default_config(self, _profile):
        return {"learning_rate": 0.1}

    def command(self, _context, _config):
        raise AssertionError("Fake runner should not execute a subprocess.")

    def capabilities(self):
        return (
            ParameterCapability("learning_rate", "exposed", "--learning-rate", "test"),
        )


class FakeRegistry:
    def __init__(self):
        self.adapter = FakeAdapter()

    def get(self, model_id):
        if model_id != self.adapter.model_id:
            raise ValueError(model_id)
        return self.adapter


class FakeRunner:
    def __init__(self, fail_first: bool = False):
        self.calls = 0
        self.fail_first = fail_first

    def run(self, *, context: TrialContext, config, **_kwargs) -> TrialResult:
        self.calls += 1
        if self.fail_first and self.calls == 1:
            return TrialResult(
                status="failed",
                validation_mse=None,
                validation_spearman=None,
                validation_mae=None,
                validation_rmse=None,
                duration_seconds=0.01,
                peak_memory_mb=1.0,
                best_step=None,
                checkpoint=None,
                command=["fake"],
                return_code=1,
                exception="intentional failure",
            )
        context.trial_dir.mkdir(parents=True, exist_ok=True)
        context.checkpoint.write_bytes(b"fake checkpoint")
        mse = float(config["learning_rate"])
        return TrialResult(
            status="completed",
            validation_mse=mse,
            validation_spearman=1.0 - mse,
            validation_mae=mse / 2,
            validation_rmse=mse**0.5,
            duration_seconds=0.01,
            peak_memory_mb=1.0,
            best_step=1,
            checkpoint=str(context.checkpoint),
            command=["fake"],
            return_code=0,
        )


class DeterministicTrial:
    def __init__(self):
        self.choices = {}

    def suggest_categorical(self, name, choices):
        self.choices[name] = list(choices)
        return list(choices)[0]

    def suggest_float(self, name, low, _high, **_kwargs):
        self.choices[name] = [low]
        return low

    def suggest_int(self, name, low, _high, **_kwargs):
        self.choices[name] = [low]
        return low


class HPOEngineTests(unittest.TestCase):
    def test_metric_parser_ignores_sentence_punctuation(self):
        parsed = _parse_intermediate(
            "Epoch 1 validation similarity MSE: 0.031020. "
            "validation_spearman=0.749931 validation_mae=0.142761",
            99,
        )
        self.assertEqual(parsed["step"], 1)
        self.assertAlmostEqual(parsed["validation_mse"], 0.031020)
        self.assertAlmostEqual(parsed["validation_spearman"], 0.749931)
        self.assertAlmostEqual(parsed["validation_mae"], 0.142761)

    def test_canonical_pair_split_has_no_reverse_leakage(self):
        graphs = [{"id": index, "nodes": list(range(3))} for index in range(8)]
        distances = {
            (left, right): float(right - left)
            for left in range(8)
            for right in range(left + 1, 8)
        }
        split = build_pair_split(graphs, distances, validation_count=5, seed=379)
        validation = set(split["validation_keys"])
        training = {
            canonical_pair_key(graphs[left]["id"], graphs[right]["id"])
            for left, right in split["training_pairs"]
        }
        self.assertTrue(validation.isdisjoint(training))

    def test_file_fingerprint_changes_with_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.zip"
            path.write_bytes(b"first")
            first = hash_file(path)
            self.assertEqual(first, hash_file(path))
            path.write_bytes(b"second")
            self.assertNotEqual(first, hash_file(path))

    def test_study_name_separates_space_versions_and_dataset_fingerprints(self):
        first = stable_study_name("simgnn", "aids700nef", "v1", "a" * 64)
        second = stable_study_name("simgnn", "aids700nef", "v2", "a" * 64)
        changed_data = stable_study_name("simgnn", "aids700nef", "v1", "b" * 64)
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, changed_data)

    def test_search_space_is_reproducible_and_dataset_adaptive(self):
        adapter = SearchSpaceRegistry().get("simgnn")
        first_trial = DeterministicTrial()
        second_trial = DeterministicTrial()
        self.assertEqual(
            adapter.suggest(first_trial, profile()),
            adapter.suggest(second_trial, profile()),
        )

        large_graph_profile = replace(
            profile(),
            train_graph_count=1200,
            node_count=DistributionStats(
                3, 40.0, 80.0, 120.0, 125.0, 160.0, 220.0, 45.0
            ),
        )
        self.assertEqual(batch_choices(large_graph_profile, (8, 16, 32, 64)), [8, 16])
        self.assertEqual(batch_choices(profile(), (8, 16, 32, 64)), [8, 16, 32])

    def test_best_config_serialization_rejects_changed_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = BestConfigRegistry(Path(directory))
            registry.save(
                dataset_profile=profile(),
                model_id="simgnn",
                search_space_version="v1",
                study_name="study",
                best_trial=2,
                validation_mse=0.1,
                validation_spearman=0.8,
                validation_mse_std=0.01,
                seeds=[379, 2026, 3407],
                hyperparameters={"learning_rate": 0.001},
                study_storage="optimization.db",
                split_seed=380,
                trial_checkpoint=None,
            )
            loaded = registry.load("unit-dataset", "simgnn", expected_fingerprint="a" * 64)
            self.assertEqual(loaded["hyperparameters"]["learning_rate"], 0.001)
            self.assertIsNone(
                registry.load("unit-dataset", "simgnn", expected_fingerprint="b" * 64)
            )

    def test_best_config_does_not_publish_machine_absolute_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = BestConfigRegistry(root / "configs")
            payload = registry.save(
                dataset_profile=profile(),
                model_id="simgnn",
                search_space_version="v1",
                study_name="study",
                best_trial=1,
                validation_mse=0.1,
                validation_spearman=0.8,
                validation_mse_std=0.0,
                seeds=[379],
                hyperparameters={"learning_rate": 0.001},
                study_storage=str(root / "optimization.db"),
                split_seed=380,
                trial_checkpoint=str(root / "trials" / "model.pt"),
            )

            self.assertFalse(Path(payload["study_storage"]).is_absolute())
            self.assertFalse(Path(payload["trial_checkpoint"]).is_absolute())

    def test_seed_everything_repeats_python_and_numpy_sequences(self):
        seed_everything(2026)
        first_python = [random.random() for _ in range(3)]
        try:
            import numpy as np
        except ImportError:
            np = None
        first_numpy = np.random.random(3).tolist() if np is not None else None
        seed_everything(2026)
        self.assertEqual(first_python, [random.random() for _ in range(3)])
        if np is not None:
            self.assertEqual(first_numpy, np.random.random(3).tolist())

    def test_failed_trial_does_not_abort_and_sqlite_study_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = FakeRunner(fail_first=True)
            optimizer = HyperparameterOptimizer(
                root=root,
                registry=FakeRegistry(),
                profiler=FakeProfiler(profile()),
                config_registry=BestConfigRegistry(root / "configs" / "optimized"),
                progress_store=ProgressStore(root / "progress"),
                runner=runner,
            )
            storage = root / "optimization.db"
            request = OptimizationRequest(
                dataset_id="unit-dataset",
                model_id="graph-fusion",
                budget="smoke",
                seed=379,
                trials=2,
                storage_path=storage,
            )
            first = optimizer.optimize(request)
            self.assertEqual(first["failed_trials"], 1)
            self.assertEqual(first["completed_trials"], 2)
            calls_after_first = runner.calls

            resumed = optimizer.optimize(replace(request, trials=3))
            self.assertGreater(resumed["completed_trials"], first["completed_trials"])
            self.assertEqual(runner.calls, calls_after_first + 1)
            self.assertTrue(storage.is_file())

    def test_multi_dataset_hpo_matrix_defaults_exclude_aids(self):
        from scripts.optimize_all import (
            NON_AIDS_DATASETS,
            ROOT,
            _display_command,
            existing_optimized_config,
        )

        self.assertNotIn("aids700nef", NON_AIDS_DATASETS)
        self.assertIn("linux", NON_AIDS_DATASETS)
        self.assertIn("imdbmulti", NON_AIDS_DATASETS)
        with patch(
            "graph_similarity_platform.hpo.service.verified_optimized_config",
            return_value={"dataset": "linux", "model": "simgnn"},
        ):
            self.assertIsNotNone(existing_optimized_config("linux", "simgnn"))
        command = _display_command(
            [str(ROOT / ".venvs" / "gnn-pyg" / "bin" / "python"), "scripts/optimize.py"]
        )
        self.assertEqual(command[0], ".venvs/gnn-pyg/bin/python")


if __name__ == "__main__":
    unittest.main()
