import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import Mock, patch

from graph_similarity_platform import training


class TrainingJobTests(unittest.TestCase):
    def setUp(self):
        training.TRAINING_JOBS.clear()
        self.model_status_patch = patch.object(
            training,
            "inspect_model",
            return_value={
                "status": "adapter_required",
                "status_label": "Adapter required",
                "missing_runtime": False,
                "missing_requirements": [],
            },
        )
        self.model_status_patch.start()

    def tearDown(self):
        self.model_status_patch.stop()
        training.TRAINING_JOBS.clear()

    def test_duplicate_running_job_is_rejected(self):
        process = Mock()
        process.poll.return_value = None
        plan = {
            "can_start": True,
            "command": "python train.py",
            "target": "checkpoints/test.pt",
            "detail": "",
            "seed": 379,
        }
        with tempfile.TemporaryDirectory(
            dir=training.BASE_DIR / "training_logs"
        ) as temporary, patch.object(
            training,
            "TRAINING_LOG_DIR",
            Path(temporary),
        ), patch.object(
            training,
            "_training_plan",
            return_value=plan,
        ), patch.object(training.subprocess, "Popen", return_value=process) as popen:
            first = training.start_training("simgnn", "aids700nef")
            self.assertEqual(first["status"], "running")

            with self.assertRaisesRegex(ValueError, "already running"):
                training.start_training("simgnn", "aids700nef")

        popen.assert_called_once()
        self.assertEqual(popen.call_args.kwargs["env"]["PYTHONUNBUFFERED"], "1")

    def test_training_plan_records_seed_and_leakage_free_protocol(self):
        plan = training._training_plan(
            training.MODEL_BY_ID["segmn"],
            "aids700nef",
            epochs=2,
            batch_size=8,
            seed=2026,
        )
        self.assertTrue(plan["can_start"])
        self.assertEqual(plan["seed"], 2026)
        self.assertIn("--seed 2026", plan["command"])
        self.assertIn("overlap is zero", plan["validation_protocol"])

    def test_training_plan_blocks_missing_model_source(self):
        missing_status = {
            "status": "missing",
            "status_label": "Code missing",
            "detail": "The third-party model folder is not installed.",
            "missing_runtime": False,
            "missing_requirements": [],
        }
        with patch.object(training, "inspect_model", return_value=missing_status):
            plan = training._training_plan(
                training.MODEL_BY_ID["simgnn"],
                "aids700nef",
            )

        self.assertFalse(plan["can_start"])
        self.assertEqual(plan["command"], "")
        self.assertIn("Model source is not ready", plan["detail"])

    def test_hpo_plan_uses_validation_only_search_and_trial_count(self):
        plan = training._training_plan(
            training.MODEL_BY_ID["simgnn"],
            "aids700nef",
            epochs=3,
            batch_size=64,
            seed=2026,
            optimize=True,
            trials=4,
        )
        self.assertTrue(plan["can_start"])
        self.assertEqual(plan["mode"], "optimization")
        self.assertEqual(plan["trials"], 4)
        self.assertIn("scripts/optimize.py", plan["command"])
        self.assertIn("--trials 4", plan["command"])
        self.assertIn("--budget standard", plan["command"])
        self.assertIn("test split is never used", plan["detail"])

    def test_hpo_result_is_parsed_from_log_tail(self):
        result = training._parse_optimization_result(
            'trial output\nHPO_RESULT={"best_validation_mse": 0.01, "promoted": true}\n'
        )
        self.assertEqual(result["best_validation_mse"], 0.01)
        self.assertTrue(result["promoted"])

    def test_persisted_hpo_with_result_is_completed_after_server_restart(self):
        with tempfile.TemporaryDirectory(
            dir=training.BASE_DIR / "training_logs"
        ) as temporary:
            root = Path(temporary)
            (root / "jobs").mkdir()
            (root / "finished.log").write_text(
                'HPO_RESULT={"best_validation_mse": 0.02, "promoted": false}\n'
            )
            (root / "jobs" / "finished.json").write_text(
                json.dumps(
                    {
                        "id": "finished",
                        "model_id": "simgnn",
                        "dataset_id": "aids700nef",
                        "status": "running",
                        "mode": "optimization",
                        "pid": 999999,
                        "log_path": str((root / "finished.log").relative_to(training.BASE_DIR)),
                        "started_at": 1.0,
                    }
                )
            )
            with patch.object(training, "TRAINING_LOG_DIR", root), patch.object(
                training, "_pid_is_running", return_value=False
            ):
                jobs = training._load_persisted_jobs()

        self.assertEqual(jobs[0]["status"], "completed")
        self.assertEqual(jobs[0]["return_code"], 0)


if __name__ == "__main__":
    unittest.main()
