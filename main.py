from pathlib import Path

import numpy as np
import pandas as pd
import typer

from enum import Enum
from sklearn.datasets import load_svmlight_file

app = typer.Typer()


class Split(str, Enum):
    train = "train"
    valid = "valid"
    test = "test"


class FeatureNormalization(str, Enum):
    log1p = "log1p"


class StorageFormat(str, Enum):
    csv = "csv"
    feather = "feather"
    parquet = "parquet"


def store(df: pd.DataFrame, path: Path, storage_format: StorageFormat):
    print(f"Store dataset: {path}")

    if storage_format == StorageFormat.csv:
        df.to_csv(path)
    elif storage_format == StorageFormat.parquet:
        df.to_parquet(path)
    elif storage_format == StorageFormat.feather:
        df.to_feather(path)
    else:
        raise ValueError("Unknown storage format")


def to_pandas(
    X: np.ndarray,
    y: np.ndarray,
    queries: np.ndarray,
    feature_prefix: str,
):
    df = pd.DataFrame(X)
    df.columns = df.columns.map(lambda x: f"{feature_prefix}{x}")
    df["relevance"] = y
    df["query_id"] = queries
    return df


@app.command()
def parse_vectors(
    istella_directory: Path,
    split: Split,
    feature_normalization: FeatureNormalization = FeatureNormalization.log1p,
    half_precision: bool = True,
    storage_format: StorageFormat = StorageFormat.feather,
    feature_prefix: str = "feature_",
):
    in_path = istella_directory / f"{split}.svm.gz"
    out_path = istella_directory / f"{split}.{storage_format}"

    assert in_path.exists(), in_path
    print(f"Parsing dataset with SVMLight format: {in_path}")
    X, y, queries = load_svmlight_file(str(in_path), query_id=True)
    X = X.todense()

    if feature_normalization == FeatureNormalization.log1p:
        print(f"Normalize features: {feature_normalization}")
        X = np.multiply(np.sign(X), np.log1p(np.abs(X)))

    if half_precision:
        print(f"Convert features and target to 16 bit half-precision")
        X = X.astype(np.float16)
        y = y.astype(np.int16)

    df = to_pandas(X, y, queries, feature_prefix)
    store(df, out_path, storage_format)


@app.command()
def train_lambdamart(
    path: Path,
):
    pass


if __name__ == "__main__":
    app()
