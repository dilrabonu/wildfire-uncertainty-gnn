from __future__ import annotations

import os
import random
import warnings

import numpy as np


# Fix OpenMP conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def _try_import_torch():
    """Attempt to import torch safely.

    Returns:
        torch module if available and healthy, otherwise None.
    """
    try:
        import torch  # type: ignore

        return torch
    except (ImportError, OSError) as exc:
        warnings.warn(
            f"PyTorch could not be imported. Continuing without torch seed setup. "
            f"Reason: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    This function always seeds Python and NumPy.
    It seeds PyTorch only if PyTorch imports successfully.

    Args:
        seed: Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch = _try_import_torch()
    if torch is None:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False