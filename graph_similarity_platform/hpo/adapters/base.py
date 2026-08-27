from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..types import DatasetProfile, TrialContext


@dataclass(frozen=True)
class ParameterCapability:
    name: str
    status: str
    binding: str
    note: str


class ModelHPOAdapter(ABC):
    model_id: str
    display_name: str
    search_space_version: str
    checkpoint_filename = "model.pt"
    resource_name = "steps"

    @abstractmethod
    def suggest(self, trial: Any, profile: DatasetProfile) -> dict[str, Any]:
        """Return only parameters bound to the real local implementation."""

    @abstractmethod
    def default_config(self, profile: DatasetProfile) -> dict[str, Any]:
        pass

    @abstractmethod
    def command(
        self,
        context: TrialContext,
        config: Mapping[str, Any],
    ) -> tuple[list[str], Path]:
        pass

    @abstractmethod
    def capabilities(self) -> tuple[ParameterCapability, ...]:
        pass

    def checkpoint_path(self, trial_dir: Path) -> Path:
        return trial_dir / self.checkpoint_filename

    def search_space_summary(self, profile: DatasetProfile) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "search_space_version": self.search_space_version,
            "default": self.default_config(profile),
            "capabilities": [capability.__dict__ for capability in self.capabilities()],
        }


def batch_choices(
    profile: DatasetProfile,
    candidates: tuple[int, ...],
    maximum: int | None = None,
) -> list[int]:
    max_nodes = profile.node_count.maximum or 1
    dataset_cap = 16 if max_nodes >= 100 else 32 if max_nodes >= 40 else 64
    if profile.train_graph_count < 350:
        dataset_cap = min(dataset_cap, 32)
    if maximum is not None:
        dataset_cap = min(dataset_cap, maximum)
    choices = sorted({value for value in candidates if value <= dataset_cap})
    return choices or [min(candidates)]


def node_cap_choices(profile: DatasetProfile) -> list[int]:
    maximum = max(4, int(profile.node_count.maximum or 4))
    q75 = max(4, int(profile.node_count.q75 or maximum))
    candidates = {min(maximum, value) for value in (12, 16, 24, 32, 48, 64)}
    candidates.add(min(maximum, q75))
    candidates.add(maximum)
    return sorted(value for value in candidates if value >= 4)
