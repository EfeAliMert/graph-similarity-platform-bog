from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from graph_similarity_platform.adapters.simgnn_utils import (
    checkpoint_hyperparameters,
    edge_index,
    normalize_graph_labels,
)
from scripts.prepare_model_datasets import (
    materialize_graphsim_dataset,
    materialized_dataset_issues,
)


class ModelAdapterPreparationTests(unittest.TestCase):
    def test_simgnn_architecture_is_recovered_from_checkpoint_shapes(self):
        state = {
            "convolution_1.lin.weight": torch.zeros((64, 7)),
            "convolution_2.lin.weight": torch.zeros((32, 64)),
            "convolution_3.lin.weight": torch.zeros((16, 32)),
            "tensor_network.bias": torch.zeros((24, 1)),
            "fully_connected_first.weight": torch.zeros((12, 40)),
        }

        values = checkpoint_hyperparameters(state, {"dropout": 0.25})

        self.assertEqual(values["filters_1"], 64)
        self.assertEqual(values["filters_2"], 32)
        self.assertEqual(values["filters_3"], 16)
        self.assertEqual(values["tensor_neurons"], 24)
        self.assertEqual(values["bottle_neck_neurons"], 12)
        self.assertEqual(values["bins"], 16)
        self.assertTrue(values["histogram"])
        self.assertEqual(values["dropout"], 0.25)

    def test_simgnn_numeric_node_ids_use_training_label_normalization(self):
        self.assertEqual(normalize_graph_labels(["0", "1", "2"]), ["0"] * 3)
        self.assertEqual(normalize_graph_labels(["C", "N", "C"]), ["C", "N", "C"])

    def test_empty_graph_has_valid_pyg_edge_index_shape(self):
        self.assertEqual(tuple(edge_index([]).shape), (2, 0))

    def test_graphsim_materialization_is_complete_and_verifiable(self):
        records = [
            {
                "id": 1,
                "split": "train",
                "nodes": [0, 1],
                "labels": ["C", "N"],
                "edges": [(0, 1)],
            },
            {
                "id": 2,
                "split": "test",
                "nodes": [0],
                "labels": ["C"],
                "edges": [],
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "sample"
            materialize_graphsim_dataset(records, root)
            self.assertEqual(materialized_dataset_issues(records, root), [])
            (root / "test" / "2.gexf").unlink()
            issues = materialized_dataset_issues(records, root)

        self.assertIn("missing test/2.gexf", issues[0])


if __name__ == "__main__":
    unittest.main()
