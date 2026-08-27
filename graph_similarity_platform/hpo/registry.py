from __future__ import annotations

from .adapters import (
    Graph2RegionHPOAdapter,
    GraphFusionHPOAdapter,
    GraphSimHPOAdapter,
    ModelHPOAdapter,
    SEGMNHPOAdapter,
    SimGNNHPOAdapter,
)


class SearchSpaceRegistry:
    def __init__(self) -> None:
        adapters = (
            SimGNNHPOAdapter(),
            GraphSimHPOAdapter(),
            SEGMNHPOAdapter(),
            GraphFusionHPOAdapter(),
            Graph2RegionHPOAdapter(),
        )
        self._adapters = {adapter.model_id: adapter for adapter in adapters}

    def get(self, model_id: str) -> ModelHPOAdapter:
        try:
            return self._adapters[model_id]
        except KeyError as exc:
            raise ValueError(f"No HPO adapter is registered for {model_id!r}.") from exc

    def model_ids(self) -> tuple[str, ...]:
        return tuple(self._adapters)


DEFAULT_REGISTRY = SearchSpaceRegistry()
