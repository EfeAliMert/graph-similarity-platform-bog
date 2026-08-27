from __future__ import annotations

import os
import random
from typing import Any


def seed_everything(seed: int, deterministic_torch: bool = True) -> dict[str, Any]:
    """Seed available RNGs without making torch a hard dependency."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    seeded = {"python": True, "numpy": False, "torch": False, "cuda": False}

    try:
        import numpy as np

        np.random.seed(seed)
        seeded["numpy"] = True
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        seeded["torch"] = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            seeded["cuda"] = True
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
    except ImportError:
        pass

    return seeded


def dataloader_worker_seed(worker_id: int, base_seed: int) -> None:
    """Deterministic worker initializer for PyTorch DataLoaders."""
    worker_seed = (int(base_seed) + int(worker_id)) % (2**32)
    random.seed(worker_seed)
    try:
        import numpy as np

        np.random.seed(worker_seed)
    except ImportError:
        pass
