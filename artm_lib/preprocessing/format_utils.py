# preprocessing/format_utils.py
from pathlib import Path

import polars as pl


# ----------------------------
# 1. Конвертация CSV → Parquet
# ----------------------------
def convert_csv_dir_to_parquet(
    csv_dir_str: str,
    parquet_dir_str: str,
    text_column: str = "text",
    overwrite: bool = False,
) -> None:
    """
    Конвертирует все CSV-файлы в директории в Parquet.
    """
    csv_dir = Path(csv_dir_str)
    parquet_dir = Path(parquet_dir_str)
    parquet_dir.mkdir(parents=True, exist_ok=True)

    if not csv_dir.exists():
        raise FileNotFoundError(f"CSV directory not found: {csv_dir_str}")

    csv_files = list(csv_dir.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {csv_dir}")

    for csv_file in sorted(csv_files):
        parquet_file = parquet_dir / f"{csv_file.stem}.parquet"
        print(f"{overwrite=}")
        if parquet_file.exists() and not overwrite:
            print(f"Skipping (already exists): {parquet_file.name}")
            continue

        print(f" Converting: {csv_file.name} → {parquet_file.name}")
        # Принудительно задаём тип текстовой колонки
        df = pl.read_csv(csv_file, dtypes={text_column: pl.Utf8})
        df.write_parquet(parquet_file)

    print("All files converted to Parquet.\n")
