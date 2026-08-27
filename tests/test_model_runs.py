import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from graph_similarity_platform import create_app
from graph_similarity_platform.graph_utils import graph_from_payload
from graph_similarity_platform import model_runs


class AutomaticModelRunTests(unittest.TestCase):
    def setUp(self):
        model_runs.MODEL_RUN_JOBS.clear()
        self.model_run_dir = model_runs.MODEL_RUN_DIR
        self.left = graph_from_payload({"edges": [[0, 1]]})
        self.right = graph_from_payload({"edges": [[0, 1], [1, 2]]})

    def tearDown(self):
        model_runs.MODEL_RUN_JOBS.clear()
        model_runs.MODEL_RUN_DIR = self.model_run_dir

    def _job(self, root: Path, model_ids: list[str], hpo_mode: str = "quick") -> dict:
        job = {
            "id": "test-job",
            "dataset_id": "aids700nef",
            "model_ids": model_ids,
            "hpo_mode": hpo_mode,
            "effective_hpo": {},
            "status": "running",
            "started_at": 1.0,
            "completed_models": 0,
            "total_models": len(model_ids),
            "failures": [],
            "progress": {},
            "_left": self.left,
            "_right": self.right,
            "_meta": {},
        }
        model_runs.MODEL_RUN_DIR = root
        model_runs.MODEL_RUN_JOBS[job["id"]] = job
        return job

    def test_selected_models_are_prepared_and_executed_sequentially(self):
        with tempfile.TemporaryDirectory() as temporary:
            job = self._job(Path(temporary), ["simgnn", "segmn"])
            calls = []

            def optimized(_job, _index, model_id, _dataset_id):
                calls.append((model_id, "hpo"))
                return {"final_training": {"status": "completed"}}

            def checkpoint(_job, _index, model_id, _dataset_id, _config):
                calls.append((model_id, "training"))

            def inference(_left, _right, model_ids, **_kwargs):
                model_id = model_ids[0]
                calls.append((model_id, "inference"))
                return [{"id": model_id, "status": "executed"}]

            with patch.object(model_runs, "_ensure_optimized_config", side_effect=optimized), patch.object(
                model_runs, "_ensure_final_checkpoint", side_effect=checkpoint
            ), patch.object(model_runs, "run_models", side_effect=inference), patch.object(
                model_runs,
                "inspect_model",
                return_value={
                    "dataset_supported": True,
                    "missing_runtime": False,
                    "missing_requirements": [],
                },
            ), patch.object(model_runs, "original_pair_matches_graphs", return_value=False):
                model_runs._run_model_job(job["id"])

        self.assertEqual(
            calls,
            [
                ("simgnn", "hpo"),
                ("simgnn", "training"),
                ("simgnn", "inference"),
                ("segmn", "hpo"),
                ("segmn", "training"),
                ("segmn", "inference"),
            ],
        )
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["progress"]["percent"], 100.0)

    def test_checkpoint_mode_skips_hpo_and_final_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            job = self._job(Path(temporary), ["simgnn"], hpo_mode="checkpoint")
            with patch.object(model_runs, "_ensure_optimized_config") as optimize, patch.object(
                model_runs, "_ensure_final_checkpoint"
            ) as train, patch.object(
                model_runs,
                "run_models",
                return_value=[{"id": "simgnn", "status": "executed"}],
            ), patch.object(
                model_runs,
                "inspect_model",
                return_value={
                    "dataset_supported": True,
                    "missing_runtime": False,
                    "missing_requirements": [],
                },
            ), patch.object(model_runs, "original_pair_matches_graphs", return_value=False):
                model_runs._run_model_job(job["id"])

        optimize.assert_not_called()
        train.assert_not_called()
        self.assertEqual(job["status"], "completed")

    def test_quick_mode_starts_two_trial_smoke_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            job = self._job(Path(temporary), ["simgnn"], hpo_mode="quick")
            config = {"study_name": "quick-study"}
            active = {
                "id": "hpo-job",
                "trials": 2,
                "budget_mode": "smoke",
            }
            with patch.object(model_runs, "running_training_job", return_value=None), patch.object(
                model_runs, "start_training", return_value=active
            ) as start, patch.object(
                model_runs, "_wait_for_training_job", return_value={"status": "completed"}
            ), patch.object(
                model_runs, "_verified_config", side_effect=[None, config]
            ), patch.object(model_runs, "_config_meets_hpo_mode", return_value=True):
                result = model_runs._ensure_optimized_config(
                    job, 0, "simgnn", "aids700nef"
                )

        self.assertIs(result, config)
        self.assertEqual(start.call_args.kwargs["trials"], 2)
        self.assertEqual(start.call_args.kwargs["budget_mode"], "smoke")
        self.assertEqual(job["effective_hpo"]["simgnn"]["effective_mode"], "smoke")

    def test_quick_mode_does_not_wait_for_longer_active_hpo_when_checkpoint_exists(self):
        with tempfile.TemporaryDirectory() as temporary:
            job = self._job(Path(temporary), ["segmn"], hpo_mode="quick")
            active = {
                "id": "research-job",
                "trials": 50,
                "budget_mode": "research",
            }
            with patch.object(
                model_runs, "running_training_job", return_value=active
            ), patch.object(
                model_runs, "_verified_config", return_value=None
            ), patch.object(
                model_runs, "_registered_checkpoint_exists", return_value=True
            ), patch.object(model_runs, "_wait_for_training_job") as wait:
                result = model_runs._ensure_optimized_config(
                    job, 0, "segmn", "aids700nef"
                )

        self.assertIsNone(result)
        wait.assert_not_called()
        self.assertEqual(
            job["effective_hpo"]["segmn"]["effective_mode"],
            "checkpoint_fallback",
        )

    def test_hpo_mode_uses_checkpoint_sidecar_without_training_logs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sidecar = root / "checkpoint.pt.hpo.json"
            sidecar.write_text('{"completed_trials": 16}')
            config = {
                "study_name": "missing-local-study",
                "final_training": {"hpo_sidecar": sidecar.name},
            }
            with patch.object(model_runs, "BASE_DIR", root):
                self.assertTrue(model_runs._config_meets_hpo_mode(config, 2))
                self.assertFalse(model_runs._config_meets_hpo_mode(config, 24))

    def test_unknown_hpo_mode_is_rejected_before_thread_start(self):
        with self.assertRaisesRegex(ValueError, "Unknown HPO mode"):
            model_runs.start_model_run(
                dataset_id="aids700nef",
                model_ids=["simgnn"],
                left=self.left,
                right=self.right,
                hpo_mode="turbo",
            )

    def test_api_forwards_selected_hpo_mode(self):
        app = create_app()
        client = app.test_client()
        with patch(
            "graph_similarity_platform.start_model_run",
            return_value={"id": "api-job", "hpo_mode": "research"},
        ) as start:
            response = client.post(
                "/api/model-runs",
                json={
                    "dataset": "aids700nef",
                    "methods": ["simgnn"],
                    "left": {"edges": [[0, 1]]},
                    "right": {"edges": [[0, 1], [1, 2]]},
                    "hpo_mode": "research",
                },
            )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(start.call_args.kwargs["hpo_mode"], "research")

    def test_identical_running_request_is_reused(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            model_runs, "MODEL_RUN_DIR", Path(temporary)
        ), patch.object(model_runs.threading.Thread, "start"):
            first = model_runs.start_model_run(
                dataset_id="aids700nef",
                model_ids=["simgnn"],
                left=self.left,
                right=self.right,
                hpo_mode="quick",
            )
            second = model_runs.start_model_run(
                dataset_id="aids700nef",
                model_ids=["simgnn"],
                left=self.left,
                right=self.right,
                hpo_mode="quick",
            )

        self.assertEqual(first["id"], second["id"])
        self.assertEqual(len(model_runs.MODEL_RUN_JOBS), 1)

    def test_model_failure_is_recorded_and_next_model_continues(self):
        with tempfile.TemporaryDirectory() as temporary:
            job = self._job(Path(temporary), ["simgnn", "segmn"])

            def optimized(_job, _index, model_id, _dataset_id):
                if model_id == "simgnn":
                    raise RuntimeError("trial process failed")
                return {}

            with patch.object(model_runs, "_ensure_optimized_config", side_effect=optimized), patch.object(
                model_runs, "_ensure_final_checkpoint"
            ), patch.object(
                model_runs,
                "run_models",
                return_value=[{"id": "segmn", "status": "executed"}],
            ) as execute, patch.object(
                model_runs,
                "inspect_model",
                return_value={
                    "dataset_supported": True,
                    "dataset_runnable": True,
                    "missing_runtime": False,
                    "missing_requirements": [],
                    "missing_files": [],
                    "checkpoints": [],
                },
            ), patch.object(model_runs, "original_pair_matches_graphs", return_value=False):
                model_runs._run_model_job(job["id"])

        self.assertEqual(job["status"], "completed")
        self.assertEqual(len(job["failures"]), 1)
        self.assertEqual(job["result"]["results"][0]["status"], "preparation_failed")
        self.assertEqual(job["result"]["results"][1]["id"], "segmn")
        execute.assert_called_once()

    def test_missing_model_files_are_reported_with_setup_path(self):
        inspection = {
            "status": "missing",
            "status_label": "Code missing",
            "detail": "The SimGNN source folder was not found.",
            "dataset_supported": True,
            "dataset_runnable": True,
            "missing_requirements": [],
            "missing_runtime": False,
            "missing_files": [],
            "checkpoints": [],
        }
        with patch.object(model_runs, "inspect_model", return_value=inspection):
            result = model_runs._failure_result(
                "simgnn",
                "aids700nef",
                "Automatic preparation failed.",
            )

        self.assertEqual(result["status"], "model_files_missing")
        self.assertEqual(result["status_label"], "Model files missing")
        self.assertIn("docs/ARTIFACT_SETUP.md", result["detail"])

    def test_missing_model_files_do_not_start_hpo(self):
        with tempfile.TemporaryDirectory() as temporary:
            job = self._job(Path(temporary), ["simgnn"])
            inspection = {
                "status": "missing",
                "status_label": "Code missing",
                "detail": "The SimGNN source folder was not found.",
                "dataset_supported": True,
                "dataset_runnable": True,
                "missing_requirements": [],
                "missing_runtime": False,
                "missing_files": [],
                "checkpoints": [],
            }
            with patch.object(
                model_runs, "inspect_model", return_value=inspection
            ), patch.object(model_runs, "_ensure_optimized_config") as optimize, patch.object(
                model_runs, "original_pair_matches_graphs", return_value=False
            ):
                model_runs._run_model_job(job["id"])

        optimize.assert_not_called()
        self.assertEqual(job["status"], "completed")
        self.assertEqual(
            job["result"]["results"][0]["status"],
            "model_files_missing",
        )

    def test_final_checkpoint_requires_matching_recorded_fingerprint(self):
        config = {
            "final_training": {
                "status": "completed",
                "checkpoint_fingerprint": "abc123",
            }
        }
        with patch.object(model_runs, "checkpoint_fingerprint", return_value="abc123"):
            self.assertTrue(model_runs._final_checkpoint_ready(config, Path("model.pt")))
        with patch.object(model_runs, "checkpoint_fingerprint", return_value="different"):
            self.assertFalse(model_runs._final_checkpoint_ready(config, Path("model.pt")))

    def test_completed_trials_switch_progress_to_confirmation(self):
        fraction, stage = model_runs._training_fraction(
            {
                "trials": 50,
                "hpo_progress": {
                    "status": "running",
                    "requested_trials": 50,
                    "elapsed_trials": 50,
                    "confirmation_run": 2,
                    "confirmation_runs": 15,
                    "current_step": 1500,
                    "resource": 3000,
                },
            },
            "hpo",
        )

        self.assertEqual(stage, "seed_confirmation")
        self.assertGreater(fraction, 0.9)
        self.assertLess(fraction, 1.0)

    def test_confirmation_detail_includes_candidate_seed_and_step(self):
        detail = model_runs._training_detail(
            {
                "hpo_progress": {
                    "confirmation_candidate": 1,
                    "confirmation_candidates": 5,
                    "confirmation_seed": 2026,
                    "current_step": 1500,
                    "resource": 3000,
                }
            },
            "SEGMN",
            "seed_confirmation",
        )

        self.assertIn("candidate 1/5", detail)
        self.assertIn("seed 2026", detail)
        self.assertIn("step 1500/3000", detail)

    def test_final_training_detail_reports_live_step(self):
        detail = model_runs._training_detail(
            {
                "command": "python train.py --steps 1000",
                "log_tail": "step=50 train_mse=0.1\nstep=100 train_mse=0.08\n",
            },
            "Graph Fusion",
            "final_training",
        )

        self.assertEqual(detail, "Graph Fusion: step 100/1000...")


if __name__ == "__main__":
    unittest.main()
