from pathlib import Path

import pandas as pd
from sklearn.datasets import load_svmlight_file


def read_svmlight_file(path: Path) -> pd.DataFrame:
    assert path.exists(), path
    print(f"Parsing dataset with SVMLight format: {path}")
    X, y, queries = load_svmlight_file(str(path), query_id=True)
    X = X.todense()

    df = pd.DataFrame(X)
    df.columns = df.columns.map(lambda x: f"feature_{x}")

    df["relevance"] = y
    df["query_id"] = queries

    return df
