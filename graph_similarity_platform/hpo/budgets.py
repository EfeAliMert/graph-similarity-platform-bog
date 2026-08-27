from __future__ import annotations

import json
from pathlib import Path

from .types import BudgetSpec


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUDGET_PATH = ROOT / "configs" / "hpo" / "budgets.json"


def load_budgets(path: Path = DEFAULT_BUDGET_PATH) -> dict[str, BudgetSpec]:
    payload = json.loads(path.read_text())
    budgets: dict[str, BudgetSpec] = {}
    for name, values in payload.items():
        budgets[name] = BudgetSpec(
            name=name,
            trials=int(values["trials"]),
            startup_trials=int(values["startup_trials"]),
            confirmation_top_k=int(values["confirmation_top_k"]),
            confirmation_seeds=tuple(int(seed) for seed in values["confirmation_seeds"]),
            resources={key: int(value) for key, value in values["resources"].items()},
            timeout_seconds=(
                None
                if values.get("timeout_seconds") is None
                else int(values["timeout_seconds"])
            ),
        )
    return budgets


def get_budget(name: str, path: Path = DEFAULT_BUDGET_PATH) -> BudgetSpec:
    budgets = load_budgets(path)
    try:
        return budgets[str(name).lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown HPO budget {name!r}; choose one of {', '.join(budgets)}."
        ) from exc
