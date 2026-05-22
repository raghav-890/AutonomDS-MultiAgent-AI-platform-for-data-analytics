"""Unit tests for orchestration state and helpers."""
import pytest


def test_infer_task_type_binary():
    """Task type correctly inferred for binary target."""
    import pandas as pd
    from app.utils.helpers import infer_task_type

    df = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
    assert infer_task_type(df, "y") == "binary_classification"


def test_infer_task_type_regression():
    """Task type correctly inferred for continuous target."""
    import pandas as pd
    import numpy as np
    from app.utils.helpers import infer_task_type

    df = pd.DataFrame({"x": np.arange(100), "price": np.random.randn(100) * 1000 + 50000})
    assert infer_task_type(df, "price") == "regression"


def test_detect_target_column():
    """Target column detection heuristic works."""
    import pandas as pd
    from app.utils.helpers import detect_target_column

    df = pd.DataFrame({"feature_1": [1, 2], "survived": [0, 1]})
    assert detect_target_column(df) == "survived"


def test_compute_dataset_stats():
    """Dataset stats computation is accurate."""
    import pandas as pd
    import numpy as np
    from app.utils.helpers import compute_dataset_stats

    df = pd.DataFrame({
        "a": [1.0, 2.0, None],
        "b": ["x", "y", "z"],
    })
    stats = compute_dataset_stats(df)
    assert stats["n_rows"] == 3
    assert stats["n_cols"] == 2
    assert stats["total_missing"] == 1


def test_safe_json_serialize():
    """JSON serialization handles numpy types."""
    import numpy as np
    from app.utils.helpers import safe_json_serialize

    obj = {"a": np.int64(1), "b": np.float32(3.14), "c": np.array([1, 2, 3])}
    result = safe_json_serialize(obj)
    assert result["a"] == 1
    assert isinstance(result["c"], list)


def test_agent_state_is_valid():
    """AgentState TypedDict accepts expected keys."""
    from app.orchestration.state import AgentState

    state: AgentState = {
        "experiment_id": "test",
        "pipeline_complete": False,
        "errors": [],
    }
    assert state["experiment_id"] == "test"
