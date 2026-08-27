from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class DistributionStats:
    count: int
    minimum: float | None
    q25: float | None
    median: float | None
    mean: float | None
    q75: float | None
    maximum: float | None
    standard_deviation: float | None


@dataclass(frozen=True)
class DatasetProfile:
    dataset_id: str
    dataset_name: str
    fingerprint: str
    profile_version: str
    target_kind: str
    target_source: str | None
    target_exact: bool
    split_strategy: str
    graph_count: int
    train_graph_count: int
    test_graph_count: int
    node_count: DistributionStats
    edge_count: DistributionStats
    density: DistributionStats
    degree: DistributionStats
    connected_components: DistributionStats
    node_label_cardinality: int
    edge_label_cardinality: int
    node_features_available: bool
    edge_labels_available: bool
    ged: DistributionStats
    normalized_ged: DistributionStats
    target_variance: float | None
    zero_target_fraction: float | None
    preprocessing: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BudgetSpec:
    name: str
    trials: int
    startup_trials: int
    confirmation_top_k: int
    confirmation_seeds: tuple[int, ...]
    resources: Mapping[str, int]
    timeout_seconds: int | None = None

    def resource_for(self, model_id: str) -> int:
        try:
            return int(self.resources[model_id])
        except KeyError as exc:
            raise ValueError(
                f"Budget {self.name!r} has no resource limit for {model_id!r}."
            ) from exc


@dataclass(frozen=True)
class TrialContext:
    root: Path
    dataset_id: str
    model_id: str
    seed: int
    split_seed: int
    resource: int
    trial_number: int
    trial_dir: Path
    checkpoint: Path
    profile: DatasetProfile


@dataclass
class TrialResult:
    status: str
    validation_mse: float | None
    validation_spearman: float | None
    validation_mae: float | None
    validation_rmse: float | None
    duration_seconds: float
    peak_memory_mb: float | None
    best_step: int | None
    checkpoint: str | None
    command: Sequence[str]
    return_code: int | None
    exception: str | None = None
    intermediate_values: list[dict[str, float | int | None]] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
