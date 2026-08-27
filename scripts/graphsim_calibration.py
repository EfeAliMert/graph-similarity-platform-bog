from __future__ import annotations

from bisect import bisect_right
import math
from typing import Any, Iterable


CALIBRATION_METHOD = "validation_isotonic_regression"


def fit_isotonic_calibration(
    raw_scores: Iterable[float],
    target_scores: Iterable[float],
    **metadata: Any,
) -> dict[str, Any]:
    import numpy as np
    from sklearn.isotonic import IsotonicRegression

    raw = np.asarray(list(raw_scores), dtype=float)
    target = np.asarray(list(target_scores), dtype=float)
    if raw.ndim != 1 or target.ndim != 1 or len(raw) != len(target):
        raise ValueError("Calibration scores must be equal-length one-dimensional arrays.")
    if len(raw) < 2:
        raise ValueError("At least two validation pairs are required for calibration.")
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(target)):
        raise ValueError("Calibration scores must be finite.")
    if np.any((target <= 0.0) | (target > 1.0)):
        raise ValueError("GraphSim calibration targets must be in (0, 1].")

    estimator = IsotonicRegression(
        y_min=1e-12,
        y_max=1.0,
        increasing=True,
        out_of_bounds="clip",
    )
    calibrated = estimator.fit_transform(raw, target)
    payload = {
        "method": CALIBRATION_METHOD,
        "version": 1,
        "x_thresholds": [float(value) for value in estimator.X_thresholds_],
        "y_thresholds": [float(value) for value in estimator.y_thresholds_],
        "fit_pair_count": int(len(raw)),
        "fit_raw_range": [float(np.min(raw)), float(np.max(raw))],
        "fit_target_range": [float(np.min(target)), float(np.max(target))],
        "fit_mse_raw": float(np.mean((raw - target) ** 2)),
        "fit_mse_calibrated": float(np.mean((calibrated - target) ** 2)),
        "fit_raw_out_of_domain_count": int(np.sum((raw <= 0.0) | (raw > 1.0))),
    }
    payload.update(metadata)
    validate_calibration(payload)
    return payload


def apply_isotonic_calibration(
    raw_score: float,
    calibration: dict[str, Any],
) -> float:
    validate_calibration(calibration)
    raw = float(raw_score)
    if not math.isfinite(raw):
        raise ValueError(f"GraphSim returned a non-finite regression output: {raw!r}")

    thresholds = [float(value) for value in calibration["x_thresholds"]]
    values = [float(value) for value in calibration["y_thresholds"]]
    if raw <= thresholds[0]:
        return values[0]
    if raw >= thresholds[-1]:
        return values[-1]

    upper = bisect_right(thresholds, raw)
    lower = upper - 1
    left_x = thresholds[lower]
    right_x = thresholds[upper]
    if right_x == left_x:
        return values[upper]
    weight = (raw - left_x) / (right_x - left_x)
    return values[lower] + weight * (values[upper] - values[lower])


def calibration_position(raw_score: float, calibration: dict[str, Any]) -> str:
    validate_calibration(calibration)
    thresholds = calibration["x_thresholds"]
    if raw_score < float(thresholds[0]):
        return "below_fit_range"
    if raw_score > float(thresholds[-1]):
        return "above_fit_range"
    return "within_fit_range"


def calibration_mse(
    raw_scores: Iterable[float],
    target_scores: Iterable[float],
    calibration: dict[str, Any],
) -> dict[str, float | int]:
    raw_values = [float(value) for value in raw_scores]
    target_values = [float(value) for value in target_scores]
    if len(raw_values) != len(target_values):
        raise ValueError("Audit scores must have equal lengths.")
    pairs = list(zip(raw_values, target_values))
    if not pairs:
        raise ValueError("At least one audit pair is required.")
    raw_mse = sum((raw - target) ** 2 for raw, target in pairs) / len(pairs)
    calibrated_mse = sum(
        (apply_isotonic_calibration(raw, calibration) - target) ** 2
        for raw, target in pairs
    ) / len(pairs)
    return {
        "audit_pair_count": len(pairs),
        "audit_mse_raw": raw_mse,
        "audit_mse_calibrated": calibrated_mse,
    }


def validate_calibration(calibration: dict[str, Any] | None) -> None:
    if not isinstance(calibration, dict):
        raise ValueError(
            "GraphSim checkpoint has no validation calibration; retrain or calibrate it."
        )
    if calibration.get("method") != CALIBRATION_METHOD:
        raise ValueError("Unsupported GraphSim output calibration method.")
    thresholds = calibration.get("x_thresholds")
    values = calibration.get("y_thresholds")
    if (
        not isinstance(thresholds, list)
        or not isinstance(values, list)
        or not thresholds
        or len(thresholds) != len(values)
    ):
        raise ValueError("GraphSim calibration thresholds are invalid.")
    numbers = [float(value) for value in thresholds + values]
    if not all(math.isfinite(value) for value in numbers):
        raise ValueError("GraphSim calibration thresholds must be finite.")
    if any(
        float(thresholds[index]) >= float(thresholds[index + 1])
        for index in range(len(thresholds) - 1)
    ):
        raise ValueError("GraphSim calibration input thresholds must increase.")
    if any(
        float(values[index]) > float(values[index + 1])
        for index in range(len(values) - 1)
    ):
        raise ValueError("GraphSim calibration output thresholds must not decrease.")
    if any(not 0.0 < float(value) <= 1.0 for value in values):
        raise ValueError("GraphSim calibration outputs must be in (0, 1].")
