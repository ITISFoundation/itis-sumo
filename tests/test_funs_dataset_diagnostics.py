import pandas as pd
import pytest

from itis_sumo.data import analyze_dataset


def test_analyze_dataset_raises_not_implemented_until_detection_logic_lands():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    with pytest.raises(NotImplementedError, match="T18ry"):
        analyze_dataset(df, input_cols=["x"], output_cols=["y"])
