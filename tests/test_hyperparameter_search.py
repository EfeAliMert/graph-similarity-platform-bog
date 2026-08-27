import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from scripts.checkpoint_provenance import (
    checkpoint_fingerprint,
    load_verified_hpo,
)
from scripts.run_hyperparameter_search import (
    ROOT,
    default_config,
    is_improvement,
    protocol_split_seed,
    protocols_are_comparable,
    trial_command,
    trial_configs,
)


class HyperparameterSearchTests(unittest.TestCase):
    def test_hpo_metadata_must_match_active_checkpoint_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            active = root / "active.pt"
            candidate = root / "candidate.pt"
            active.write_bytes(b"matching weights")
            candidate.write_bytes(b"matching weights")
            sidecar = Path(str(active) + ".hpo.json")
            sidecar.write_text(
                json.dumps(
                    {
                        "active_checkpoint": "active.pt",
                        "study_id": "verified-study",
                        "best_trial": {"checkpoint": "candidate.pt"},
                    }
                )
            )

            payload, status = load_verified_hpo(active, root)
            self.assertEqual(status, "verified_checkpoint")
            self.assertEqual(payload["study_id"], "verified-study")

            active.write_bytes(b"new weights")
            payload, status = load_verified_hpo(active, root)
            self.assertEqual(status, "stale_checkpoint")
            self.assertEqual(payload, {})

    def test_tensorflow_checkpoint_fingerprint_uses_weight_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.ckpt"
            second = root / "second.ckpt"
            for suffix, content in (
                (".data-00000-of-00001", b"weights"),
                (".index", b"index"),
                (".meta", b"graph"),
            ):
                Path(str(first) + suffix).write_bytes(content)
                Path(str(second) + suffix).write_bytes(content)
            self.assertEqual(
                checkpoint_fingerprint(first),
                checkpoint_fingerprint(second),
            )

    def test_validation_protocol_must_match_before_promotion(self):
        incumbent = {"pair_split": {"seed": 10, "split_sha256": "same"}}
        candidate = {"pair_split": {"seed": 11, "split_sha256": "same"}}
        mismatch = {"pair_split": {"seed": 10, "split_sha256": "other"}}
        self.assertTrue(
            protocols_are_comparable("segmn", incumbent, candidate, True)
        )
        self.assertFalse(
            protocols_are_comparable("segmn", incumbent, mismatch, True)
        )
        self.assertTrue(protocols_are_comparable("segmn", {}, {}, False))
        self.assertEqual(protocol_split_seed("segmn", incumbent, 379), 10)

    def test_promotion_requires_validation_improvement(self):
        self.assertTrue(is_improvement(0.1, None))
        self.assertTrue(is_improvement(0.1, 0.2))
        self.assertFalse(is_improvement(0.2, 0.2))
        self.assertFalse(is_improvement(0.3, 0.2))
        self.assertFalse(is_improvement(float("nan"), 0.2))

    def test_trial_configs_are_deterministic_and_include_default_first(self):
        first = trial_configs("simgnn", 64, 6, 2026)
        second = trial_configs("simgnn", 64, 6, 2026)
        self.assertEqual(first, second)
        self.assertEqual(first[0], default_config("simgnn", 64))
        self.assertEqual(len(first), 6)
        self.assertEqual(len({tuple(sorted(row.items())) for row in first}), 6)

    def test_simgnn_trial_command_binds_validation_and_hyperparameters(self):
        args = SimpleNamespace(
            model="simgnn",
            dataset="aids700nef",
            budget=4,
            seed=2026,
            split_seed=2026,
        )
        config = default_config("simgnn", 64)
        command, _cwd = trial_command(args, config, ROOT / "training_logs/test/model.pt")
        joined = " ".join(command)
        self.assertIn("--validation-graphs", command)
        self.assertIn("--learning-rate 0.001", joined)
        self.assertIn("--dropout 0.5", joined)
        self.assertIn("--weight-decay 0.0005", joined)
        self.assertIn("--seed 2026", joined)


if __name__ == "__main__":
    unittest.main()
