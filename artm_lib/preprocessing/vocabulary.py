from typing import Callable
from pathlib import Path

import polars as pl


# ---------------------------------
# 2. Построение словаря и doc_index
# ---------------------------------
def simple_tokenizer(x):
    return x.strip().split()


def build_vocab_and_index_from_parquet(
    parquet_dir_str: str,
    text_column: str = "Description",
    tokenizer: Callable[[str], list[str]] = simple_tokenizer,
    min_df: int = 1,
    max_df: float = 1.0,
) -> tuple[dict, list[tuple[str, int]]]:
    """
    Проходит по всем Parquet-файлам, строит частотный словарь и индекс документов.
    Возвращает:
        token_to_id: Dict[str, int]
        doc_index: List[(parquet_file_path, line_index)]
    """
    from collections import Counter

    parquet_dir = Path(parquet_dir_str)
    token_freq = Counter()
    doc_index = []

    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"No Parquet files found in {parquet_dir}")

    for pq_file in parquet_files:
        print(f" Scanning: {pq_file.name}")
        # Читаем только текстовую колонку
        df = pl.read_parquet(pq_file, columns=[text_column])
        texts = df[text_column].to_list()
        for line_idx, text in enumerate(texts):
            doc_index.append((str(pq_file), line_idx))
            if text is None:
                continue
            tokens = tokenizer(str(text))
            token_freq.update(tokens)

    # Фильтрация
    total_docs = len(doc_index)
    min_count = min_df
    max_count = int(max_df * total_docs) if max_df < 1.0 else float("inf")

    filtered_tokens = [
        token for token, freq in token_freq.items() if min_count <= freq <= max_count
    ]
    filtered_tokens.sort()
    token_to_id = {token: i for i, token in enumerate(filtered_tokens)}

    print(f"Vocab size: {len(token_to_id)} | Documents: {len(doc_index)}\n")
    return token_to_id, doc_index
