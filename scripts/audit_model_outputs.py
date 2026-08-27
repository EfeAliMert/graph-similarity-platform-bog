from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graph_similarity_platform.data import (  # noqa: E402
    list_original_graphs,
    load_original_pair,
    pair_ground_truth,
)
from graph_similarity_platform.graph_utils import graph_from_payload  # noqa: E402
from graph_similarity_platform.models.real_models import MODELS, run_models  # noqa: E402


REPORT_JSON = ROOT / "reports" / "model_output_audit.json"
REPORT_MARKDOWN = ROOT / "reports" / "model_output_audit.md"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit model-output binding, scale conversion, symmetry, and identity behavior."
    )
    parser.add_argument("--dataset", default="aids700nef")
    parser.add_argument("--left")
    parser.add_argument("--right")
    args = parser.parse_args()

    choices = list_original_graphs(args.dataset)
    left_member = args.left or choices["train"][0]["member"]
    right_member = args.right or choices["test"][0]["member"]
    payload = audit_pair(args.dataset, left_member, right_member)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True))
    REPORT_MARKDOWN.write_text(render_markdown(payload))
    print(f"technical_integrity={payload['technical_integrity_passed']}/{len(payload['models'])}")
    print(f"json={REPORT_JSON.relative_to(ROOT)}")
    print(f"markdown={REPORT_MARKDOWN.relative_to(ROOT)}")


def audit_pair(dataset_id: str, left_member: str, right_member: str) -> dict[str, Any]:
    forward_pair = load_original_pair(dataset_id, left_member, right_member)
    reverse_pair = load_original_pair(dataset_id, right_member, left_member)
    identity_pair = load_original_pair(dataset_id, left_member, left_member)
    model_ids = [model["id"] for model in MODELS]
    forward = _run_pair(dataset_id, forward_pair, model_ids)
    reverse = _run_pair(dataset_id, reverse_pair, model_ids)
    identity = _run_pair(dataset_id, identity_pair, model_ids)
    left = graph_from_payload(forward_pair["left"], name="Graph A")
    right = graph_from_payload(forward_pair["right"], name="Graph B")
    reference = pair_ground_truth(
        dataset_id,
        left_member,
        right_member,
        left.node_count,
        right.node_count,
    )
    rows = []
    for model_id in model_ids:
        rows.append(
            _audit_row(
                next(row for row in forward if row["id"] == model_id),
                next(row for row in reverse if row["id"] == model_id),
                next(row for row in identity if row["id"] == model_id),
                reference,
                0.5 * (left.node_count + right.node_count),
            )
        )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": dataset_id,
        "left_graph": left_member,
        "right_graph": right_member,
        "reference_ged": (reference or {}).get("distance"),
        "reference_similarity": (reference or {}).get("similarity"),
        "reference_exact": (reference or {}).get("exact"),
        "reference_kind": (reference or {}).get("reference_kind"),
        "reference_source": (reference or {}).get("source"),
        "technical_integrity_passed": sum(row["technical_integrity"] for row in rows),
        "warning": (
            "Technical integrity verifies execution, input binding, bounds, and GED-scale "
            "conversion. It does not certify checkpoint accuracy."
        ),
        "models": rows,
    }


def _run_pair(dataset_id: str, pair: dict[str, Any], model_ids: list[str]) -> list[dict[str, Any]]:
    left = graph_from_payload(pair["left"], name="Graph A")
    right = graph_from_payload(pair["right"], name="Graph B")
    return run_models(
        left,
        right,
        model_ids,
        dataset_id=dataset_id,
        meta=pair["meta"],
    )


def _audit_row(
    forward: dict[str, Any],
    reverse: dict[str, Any],
    identity: dict[str, Any],
    reference: dict[str, Any] | None,
    graph_size: float,
) -> dict[str, Any]:
    metrics = forward.get("adapter_metrics") or {}
    predicted_ged = _finite_number(metrics.get("predicted_ged"))
    canonical = _finite_number(forward.get("canonical_similarity"))
    expected = (
        math.exp(-predicted_ged / max(float(graph_size), 1.0))
        if predicted_ged is not None and predicted_ged >= 0.0
        else None
    )
    conversion_consistent = (
        canonical is None and expected is None
    ) or (
        canonical is not None
        and expected is not None
        and math.isclose(canonical, expected, rel_tol=1e-7, abs_tol=1e-9)
    )
    native = _finite_number(forward.get("model_score"))
    reverse_canonical = _finite_number(reverse.get("canonical_similarity"))
    identity_canonical = _finite_number(identity.get("canonical_similarity"))
    reference_ged = _finite_number((reference or {}).get("distance"))
    reference_exact = (reference or {}).get("exact") is not False
    technical_integrity = bool(
        forward.get("status") == "executed"
        and reverse.get("status") == "executed"
        and identity.get("status") == "executed"
        and forward.get("checkpoint_loaded")
        and forward.get("architecture_loaded")
        and forward.get("input_matches_dataset_pair") is True
        and native is not None
        and 0.0 <= native <= 1.0
        and conversion_consistent
    )
    return {
        "model_id": forward["id"],
        "model_name": forward["name"],
        "technical_integrity": technical_integrity,
        "status": forward.get("status"),
        "native_output": native,
        "comparable_similarity": canonical,
        "predicted_ged": predicted_ged,
        "reference_ged_error": (
            abs(predicted_ged - reference_ged)
            if predicted_ged is not None and reference_ged is not None
            else None
        ),
        "exact_ged_error": (
            abs(predicted_ged - reference_ged)
            if reference_exact
            and predicted_ged is not None
            and reference_ged is not None
            else None
        ),
        "conversion_consistent": conversion_consistent,
        "reverse_comparable_similarity": reverse_canonical,
        "symmetry_gap": (
            abs(canonical - reverse_canonical)
            if canonical is not None and reverse_canonical is not None
            else None
        ),
        "identity_comparable_similarity": identity_canonical,
        "identity_predicted_ged": _finite_number(
            (identity.get("adapter_metrics") or {}).get("predicted_ged")
        ),
        "score_semantics": forward.get("score_semantics"),
        "checkpoint": forward.get("selected_checkpoint"),
        "seed": metrics.get("seed"),
        "target": metrics.get("target"),
        "hyperparameters": metrics.get("hyperparameters"),
        "hpo": metrics.get("hpo"),
    }


def _finite_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Model Output Audit",
        "",
        payload["warning"],
        "",
        f"- Dataset: `{payload['dataset_id']}`",
        f"- Pair: `{payload['left_graph']}` vs `{payload['right_graph']}`",
        f"- GED reference: `{payload['reference_ged']}`",
        f"- Reference kind: `{payload['reference_kind']}`",
        (
            f"- Technical integrity: `{payload['technical_integrity_passed']}/"
            f"{len(payload['models'])}`"
        ),
        "",
        "| Model | Integrity | Native | Comparable | Pred. GED | GED error | Symmetry gap | Identity similarity |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["models"]:
        lines.append(
            "| {model} | {integrity} | {native} | {comparable} | {ged} | "
            "{error} | {symmetry} | {identity} |".format(
                model=row["model_name"],
                integrity="pass" if row["technical_integrity"] else "review",
                native=_display(row["native_output"]),
                comparable=_display(row["comparable_similarity"]),
                ged=_display(row["predicted_ged"]),
                error=_display(row["reference_ged_error"]),
                symmetry=_display(row["symmetry_gap"]),
                identity=_display(row["identity_comparable_similarity"]),
            )
        )
    lines.extend(
        [
            "",
            "Identity and symmetry columns are checkpoint-behavior diagnostics, not "
            "technical execution criteria. Weak values indicate model-fit limitations.",
            "",
        ]
    )
    return "\n".join(lines)


def _display(value: Any) -> str:
    return "-" if value is None else f"{float(value):.6f}"


if __name__ == "__main__":
    main()
