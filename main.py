from pathlib import Path

import typer

from src.svmlight import read_svmlight_file

app = typer.Typer()


@app.command()
def parse_svm_split(
        base_dir: Path,
        split: str,
):
    in_path = base_dir / f"{split}.svm.gz"
    out_path = base_dir / f"{split}.parquet"

    df = read_svmlight_file(in_path)
    df.to_parquet(out_path)


if __name__ == "__main__":
    app()
