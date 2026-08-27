import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import zipfile

from graph_similarity_platform import create_app
from graph_similarity_platform import data as dataset_data
from graph_similarity_platform import training
from scripts import universal_dataset


def gexf(label: str) -> bytes:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<gexf xmlns="http://www.gexf.net/1.2draft" version="1.2">
  <graph mode="static" defaultedgetype="undirected">
    <nodes>
      <node id="0" label="{label}" />
      <node id="1" label="{label}" />
    </nodes>
    <edges><edge id="0" source="0" target="1" /></edges>
  </graph>
</gexf>
""".encode()


def graph_archive(unsafe: bool = False) -> io.BytesIO:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        prefix = "../" if unsafe else "custom/"
        archive.writestr(f"{prefix}train/0.gexf", gexf("C"))
        archive.writestr(f"{prefix}train/1.gexf", gexf("N"))
        archive.writestr(f"{prefix}test/2.gexf", gexf("O"))
        archive.writestr(f"{prefix}test/3.gexf", gexf("F"))
    payload.seek(0)
    return payload


class DatasetUploadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.upload_root = Path(self.temporary.name) / "uploaded_datasets"
        self.patch = patch.object(dataset_data, "UPLOADED_DATASET_DIR", self.upload_root)
        self.universal_root_patch = patch.object(
            universal_dataset,
            "UPLOADED_ROOT",
            self.upload_root,
        )
        self.derived_root_patch = patch.object(
            universal_dataset,
            "DERIVED_ROOT",
            Path(self.temporary.name) / "derived_training",
        )
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
        self.patch.start()
        self.universal_root_patch.start()
        self.derived_root_patch.start()
        self.model_status_patch.start()
        dataset_data._cached_ground_truth.cache_clear()
        self.client = create_app().test_client()

    def tearDown(self):
        dataset_data._cached_ground_truth.cache_clear()
        self.model_status_patch.stop()
        self.derived_root_patch.stop()
        self.universal_root_patch.stop()
        self.patch.stop()
        self.temporary.cleanup()

    def test_upload_is_persisted_registered_and_trainable(self):
        response = self.client.post(
            "/api/datasets/upload",
            data={
                "name": "Research Graphs",
                "dataset_id": "research-graphs",
                "domain": "Test molecules",
                "archive": (graph_archive(), "graphs.zip"),
                "ground_truth": (
                    io.BytesIO(b"left,right,ged\n0,1,1\n2,0,2\n3,1,3\n"),
                    "ged.csv",
                ),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 201)
        uploaded = response.get_json()["dataset"]
        self.assertEqual(uploaded["id"], "research-graphs")
        self.assertEqual(uploaded["train_graphs"], 2)
        self.assertEqual(uploaded["test_graphs"], 2)
        self.assertTrue(uploaded["training_ready"])
        self.assertTrue((self.upload_root / "research-graphs" / "dataset.zip").exists())
        self.assertTrue((self.upload_root / "research-graphs" / "ged.json").exists())
        self.assertTrue((self.upload_root / "research-graphs" / "manifest.json").exists())

        catalog = self.client.get("/api/datasets").get_json()["datasets"]
        self.assertIn("research-graphs", {dataset["id"] for dataset in catalog})

        graphs = self.client.get("/api/datasets/research-graphs/graphs").get_json()
        self.assertEqual(len(graphs["train"]), 2)
        self.assertEqual(len(graphs["test"]), 2)

        pair = self.client.get(
            "/api/datasets/research-graphs/pair",
            query_string={
                "left": graphs["train"][0]["member"],
                "right": graphs["test"][0]["member"],
            },
        )
        self.assertEqual(pair.status_code, 200)
        self.assertEqual(pair.get_json()["meta"]["dataset_id"], "research-graphs")

        training = self.client.get(
            "/api/training",
            query_string={"dataset": "research-graphs"},
        ).get_json()
        plans = {plan["id"]: plan for plan in training["plans"]}
        self.assertEqual(len(plans), 5)
        self.assertTrue(all(plan["can_start"] for plan in plans.values()))
        self.assertTrue(all("research-graphs" in plan["target"] for plan in plans.values()))
        self.assertTrue(
            all(
                plan["target_source"] == "user-provided unverified GED"
                and "not been independently verified" in plan["detail"]
                for plan in plans.values()
            )
        )

    def test_upload_without_ged_can_train_with_structural_proxy(self):
        response = self.client.post(
            "/api/datasets/upload",
            data={
                "name": "Preview Graphs",
                "archive": (graph_archive(), "graphs.zip"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 201)
        uploaded = response.get_json()["dataset"]
        self.assertFalse(uploaded["training_ready"])
        training = self.client.get(
            "/api/training",
            query_string={"dataset": uploaded["id"]},
        ).get_json()
        self.assertTrue(all(plan["can_start"] for plan in training["plans"]))
        self.assertTrue(
            all(uploaded["id"] in plan["command"] for plan in training["plans"])
        )
        self.assertTrue(
            all(
                plan["target_source"] == "structural GED proxy"
                and "not a GED benchmark" in plan["detail"]
                for plan in training["plans"]
            )
        )

    def test_incomplete_ged_uses_proxy_for_training(self):
        response = self.client.post(
            "/api/datasets/upload",
            data={
                "name": "Partial GED Graphs",
                "archive": (graph_archive(), "graphs.zip"),
                "ground_truth": (
                    io.BytesIO(b"left,right,ged\n0,1,1\n"),
                    "ged.csv",
                ),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 201)
        uploaded = response.get_json()["dataset"]
        self.assertEqual(uploaded["ground_truth_available"], 1)
        self.assertFalse(uploaded["training_ready"])
        training = self.client.get(
            "/api/training",
            query_string={"dataset": uploaded["id"]},
        ).get_json()
        self.assertTrue(all(plan["can_start"] for plan in training["plans"]))
        self.assertTrue(
            all(
                plan["target_source"] == "structural GED proxy"
                for plan in training["plans"]
            )
        )
        distances, target = universal_dataset.ensure_training_distances(uploaded["id"])
        self.assertTrue(distances)
        self.assertFalse(target["exact"])
        self.assertEqual(target["target_source"], "derived structural GED proxy")
        self.assertIn("did not cover", target["reason"])

    def test_unsafe_archive_member_is_rejected(self):
        response = self.client.post(
            "/api/datasets/upload",
            data={
                "name": "Unsafe Graphs",
                "archive": (graph_archive(unsafe=True), "graphs.zip"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsafe archive member path", response.get_json()["error"])
        self.assertFalse((self.upload_root / "unsafe-graphs").exists())


if __name__ == "__main__":
    unittest.main()
