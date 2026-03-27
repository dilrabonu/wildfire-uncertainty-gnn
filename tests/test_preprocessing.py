from __future__ import annotations

import numpy as np

from wildfire_gnn.data.preprocessing import summarize_array


def test_summarize_array_basic() -> None:
    arr = np.array([[1, 2], [3, 4]], dtype=float)
    stats = summarize_array(arr)

    assert stats["min"] == 1.0
    assert stats["max"] == 4.0
    assert stats["count"] == 4