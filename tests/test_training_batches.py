import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

try:
    import torch
    from torch_geometric.data import Data
    from torch_geometric.nn import global_mean_pool

    from scripts.train_graph2region_universal import G2R, build_model_args, score_pairs
    from scripts.train_graphsim_compat import rebalance_training_triples
    from segmn_universal import AIDS_ATOM_TYPES, build_args, record_to_segmn, transform_pair
    from universal_pyg import record_to_data
except ModuleNotFoundError as exc:
    torch = None
    OPTIONAL_IMPORT_ERROR = str(exc)
else:
    OPTIONAL_IMPORT_ERROR = ""


class DummyRegionModel:
    def __call__(self, x, _edge_index):
        return x, torch.zeros_like(x)

    def union(self, regions, _offsets, batch):
        pooled = global_mean_pool(regions, batch)
        return torch.stack((pooled, pooled), dim=0)

    def predict_norm_ged(self, left, right):
        return torch.exp(-torch.abs(left - right).sum(dim=-1))


@unittest.skipIf(torch is None, f"optional GNN runtime unavailable: {OPTIONAL_IMPORT_ERROR}")
class Graph2RegionBatchTests(unittest.TestCase):
    def test_corrected_positional_encoding_is_repeatable(self):
        args = build_model_args("aids700nef", max_nodes=10, seed=379)
        model = G2R(args).eval()
        x = torch.ones((3, 1))
        edge_index = torch.tensor(
            [[0, 1, 1, 2], [1, 0, 2, 1]],
            dtype=torch.long,
        )

        _, first = model(x, edge_index)
        _, second = model(x, edge_index)

        self.assertTrue(torch.equal(first, second))

    def test_corrected_ged_geometry_maps_identical_regions_to_one(self):
        args = build_model_args("aids700nef", max_nodes=10, seed=379)
        model = G2R(args)
        regions = torch.tensor([[0.2, -0.1, 0.5, 0.0, 0.3, -0.2, 0.1, 0.4]])

        score = model.predict_norm_ged(regions, regions)

        self.assertAlmostEqual(float(score.item()), 1.0, places=7)

    def test_score_pairs_keeps_one_prediction_per_pair(self):
        first = Data(
            x=torch.tensor([[0.0], [0.0]]),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            graph_id=1,
        )
        second = Data(
            x=torch.tensor([[1.0], [1.0]]),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            graph_id=2,
        )
        predictions, targets = score_pairs(
            DummyRegionModel(),
            [first, second],
            [(0, 0), (0, 1)],
            {(1, 1): 0.0, (1, 2): 2.0},
            torch.device("cpu"),
        )

        self.assertEqual(tuple(predictions.shape), (2,))
        self.assertEqual(tuple(targets.shape), (2,))
        self.assertAlmostEqual(float(predictions[0]), 1.0, places=6)
        self.assertLess(float(predictions[1]), float(predictions[0]))


@unittest.skipIf(torch is None, f"optional GNN runtime unavailable: {OPTIONAL_IMPORT_ERROR}")
class GraphSimSamplingTests(unittest.TestCase):
    def test_inductive_graph_ids_enable_distinct_zero_pair_oversampling(self):
        class Graph:
            def __init__(self, graph_id):
                self.nxgraph = type("NxGraph", (), {"graph": {"gid": graph_id}})()

        class Sampler:
            def __init__(self, triples):
                self.li = triples
                self.idx = 0

            def _shuffle(self):
                return None

        first = Graph(1)
        second = Graph(2)
        third = Graph(3)
        model = type(
            "Model",
            (),
            {
                "train_triples": Sampler(
                    [
                        (first, first, 1.0),
                        (first, second, 1.0),
                        (second, first, 1.0),
                        (first, third, 0.5),
                    ]
                )
            },
        )()

        audit = rebalance_training_triples(model, seed=379, zero_fraction=0.75)

        self.assertEqual(audit["exact_zero_source_pairs"], 2)
        self.assertEqual(audit["exact_zero_pairs_after_oversampling"], 3)
        self.assertEqual(len(model.train_triples.li), 5)


@unittest.skipIf(torch is None, f"optional GNN runtime unavailable: {OPTIONAL_IMPORT_ERROR}")
class SegmnFeatureTests(unittest.TestCase):
    def test_isomorphic_labeled_graphs_share_canonical_segmn_tensors(self):
        args = build_args(
            torch.device("cpu"),
            node_cap=4,
            edge_cap=4,
            max_degree=3,
            label_vocabulary=["C", "O"],
            edge_feature_mode="sum",
            architecture_profile="compact",
            canonical_node_order=True,
        )
        first = {
            "id": 1,
            "nodes": ["a", "b", "c", "d"],
            "edges": [("a", "b"), ("b", "c"), ("b", "d")],
            "labels": ["O", "C", "C", "O"],
        }
        second = {
            "id": 2,
            "nodes": ["w", "x", "y", "z"],
            "edges": [("z", "w"), ("x", "z"), ("y", "z")],
            "labels": ["C", "O", "O", "C"],
        }

        left = record_to_segmn(first, args)
        right = record_to_segmn(second, args)

        for field in ("x", "x1", "edge_index", "edgeindex1", "h", "f"):
            self.assertTrue(torch.equal(getattr(left, field), getattr(right, field)), field)

    def test_isomorphic_graphs_share_canonical_pyg_tensors(self):
        first = {
            "id": 1,
            "member": "train/1.gexf",
            "nodes": ["a", "b", "c", "d"],
            "edges": [("a", "b"), ("b", "c"), ("b", "d")],
            "labels": ["O", "C", "C", "O"],
        }
        second = {
            "id": 2,
            "member": "test/2.gexf",
            "nodes": ["w", "x", "y", "z"],
            "edges": [("z", "w"), ("x", "z"), ("y", "z")],
            "labels": ["C", "O", "O", "C"],
        }

        left = record_to_data(first, feature_mode="degree", canonical_order=True)
        right = record_to_data(second, feature_mode="degree", canonical_order=True)

        self.assertTrue(torch.equal(left.x, right.x))
        self.assertTrue(torch.equal(left.edge_index, right.edge_index))

    def test_aids_profile_keeps_atom_and_degree_features(self):
        args = build_args(
            torch.device("cpu"),
            node_cap=4,
            edge_cap=4,
            max_degree=6,
            label_vocabulary=AIDS_ATOM_TYPES,
            edge_feature_mode="sum",
            architecture_profile="aids-original",
        )
        record = {
            "id": 1,
            "nodes": ["a", "b"],
            "edges": [("a", "b")],
            "labels": ["C", "O"],
        }

        graph = record_to_segmn(record, args)

        self.assertEqual(args.D, 36)
        self.assertEqual(args.x_size, 36)
        self.assertEqual(tuple(graph.x.shape), (2, 36))
        self.assertEqual(tuple(graph.x1.shape), (1, 36))
        self.assertEqual(int(graph.x[0, :29].argmax()), AIDS_ATOM_TYPES.index("C"))
        self.assertEqual(int(graph.x[1, :29].argmax()), AIDS_ATOM_TYPES.index("O"))
        self.assertTrue(torch.equal(graph.x1[0], graph.x[0] + graph.x[1]))

    def test_linux_profile_matches_original_degree_dimensions(self):
        args = build_args(
            torch.device("cpu"),
            node_cap=10,
            edge_cap=13,
            line_edge_cap=30,
            max_degree=7,
            edge_feature_mode="concat",
            architecture_profile="linux-original",
        )
        record = {
            "id": 1,
            "nodes": ["a", "b", "c"],
            "edges": [("a", "b"), ("b", "c")],
            "labels": ["0", "0", "0"],
        }

        graph = record_to_segmn(record, args)

        self.assertEqual(args.D, 8)
        self.assertEqual(args.x_size, 16)
        self.assertEqual(args.embedding_size, 128)
        self.assertEqual(args.n_heads, 4)
        self.assertEqual(tuple(graph.x.shape), (3, 8))
        self.assertEqual(tuple(graph.x1.shape), (2, 16))

    def test_edgeless_graph_transforms_without_empty_batch_crash(self):
        args = build_args(
            torch.device("cpu"),
            node_cap=4,
            edge_cap=4,
            max_degree=3,
            architecture_profile="compact",
        )
        isolated = {
            "id": 1,
            "nodes": ["a", "b"],
            "edges": [],
            "labels": ["0", "0"],
        }

        graph = record_to_segmn(isolated, args)
        payload = transform_pair(graph, graph, args, target=0.0)

        self.assertEqual(tuple(graph.x1.shape), (0, int(args.x_size)))
        self.assertEqual(tuple(payload["g0"]["x1"].shape), (1, int(args.n_max_edges), int(args.x_size)))
        self.assertFalse(bool(payload["g0"]["mask_x1"].any()))


if __name__ == "__main__":
    unittest.main()
