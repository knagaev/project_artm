# prepare_artm_dataset.py
import os
from pathlib import Path
from typing import Callable, List, Tuple
import polars as pl
import torch

# from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix

# from collections import Counter
import numpy as np


class ARTMCollator:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def __call__(self, batch):
        # batch: List[(doc_id, List[token_id])]
        doc_ids = [item[0] for item in batch]
        list_of_token_ids = [item[1] for item in batch]

        # Если батч пуст
        if not list_of_token_ids or all(len(x) == 0 for x in list_of_token_ids):
            bow_matrix = coo_matrix((0, self.vocab_size), dtype=np.int32).tocsr()
            return doc_ids, bow_matrix

        # 1. Линеаризуем токены и строим массив doc_indices
        all_tokens = []
        doc_indices = []
        for i, token_ids in enumerate(list_of_token_ids):
            if token_ids:
                all_tokens.extend(token_ids)
                doc_indices.extend([i] * len(token_ids))

        if not all_tokens:
            bow_matrix = coo_matrix(
                (len(batch), self.vocab_size), dtype=np.int32
            ).tocsr()
            return doc_ids, bow_matrix

        all_tokens = np.array(all_tokens, dtype=np.int32)
        doc_indices = np.array(doc_indices, dtype=np.int32)

        # 2. Фильтруем токены вне словаря (защита)
        mask = (all_tokens >= 0) & (all_tokens < self.vocab_size)
        all_tokens = all_tokens[mask]
        doc_indices = doc_indices[mask]

        # 3. Создаём разреженную матрицу
        data = np.ones_like(all_tokens, dtype=np.int32)
        bow_matrix = coo_matrix(
            (data, (doc_indices, all_tokens)),
            shape=(len(batch), self.vocab_size),
            dtype=np.int32,
        ).tocsr()

        return doc_ids, bow_matrix


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
        if parquet_file.exists() and not overwrite:
            print(f"Skipping (already exists): {parquet_file.name}")
            continue

        print(f" Converting: {csv_file.name} → {parquet_file.name}")
        # Принудительно задаём тип текстовой колонки
        df = pl.read_csv(csv_file, dtypes={text_column: pl.Utf8})
        df.write_parquet(parquet_file)

    print("All files converted to Parquet.\n")


# ---------------------------------
# 2. Построение словаря и doc_index
# ---------------------------------
def simple_tokenizer(x):
    return x.strip().split()


def build_vocab_and_index_from_parquet(
    parquet_dir_str: str,
    text_column: str = "text",
    tokenizer: Callable[[str], List[str]] = simple_tokenizer,
    min_df: int = 1,
    max_df: float = 1.0,
) -> Tuple[dict, List[Tuple[str, int]]]:
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


# ----------------------------
# 3. ARTMDatasetParquet
# ----------------------------
class ARTMDatasetParquet(torch.utils.data.Dataset):
    def __init__(
        self,
        doc_index: List[Tuple[str, int]],
        token_to_id: dict,
        text_column: str,
        tokenizer: Callable[[str], List[str]],
        cache_dataframes: bool = True,
    ):
        self.doc_index = doc_index
        self.token_to_id = token_to_id
        self.text_column = text_column
        self.tokenizer = tokenizer
        self.cache_dataframes = cache_dataframes
        self._df_cache = {}

    def _get_df(self, file_path: str):
        if self.cache_dataframes:
            if file_path not in self._df_cache:
                self._df_cache[file_path] = pl.read_parquet(
                    file_path, columns=[self.text_column]
                )
            return self._df_cache[file_path]
        else:
            return pl.read_parquet(file_path, columns=[self.text_column])

    def __len__(self):
        return len(self.doc_index)

    def __getitem__(self, idx):
        file_path, line_idx = self.doc_index[idx]
        df = self._get_df(file_path)
        text = df.row(line_idx)[0]  # только одна колонка
        if text is None:
            text = ""
        tokens = self.tokenizer(str(text))
        token_ids = [self.token_to_id[t] for t in tokens if t in self.token_to_id]
        return idx, token_ids


# ----------------------------
# 5. Пример использования
# ----------------------------
def custom_tokenizer(text: str) -> List[str]:
    return text.lower().split()


if __name__ == "__main__":
    # Конфигурация
    CSV_DIR = "./data/csvs"
    PARQUET_DIR = "./data/parquets"
    TEXT_COLUMN = "Description"
    MIN_DF = 5
    MAX_DF = 0.95

    # Шаг 1: Конвертация
    convert_csv_dir_to_parquet(
        csv_dir_str=CSV_DIR,
        parquet_dir_str=PARQUET_DIR,
        text_column=TEXT_COLUMN,
        overwrite=False,
    )

    # Шаг 2: Построение словаря и индекса
    token_to_id, doc_index = build_vocab_and_index_from_parquet(
        parquet_dir_str=PARQUET_DIR,
        text_column=TEXT_COLUMN,
        tokenizer=custom_tokenizer,
        min_df=MIN_DF,
        max_df=MAX_DF,
    )

    # Шаг 3: Создание датасета
    dataset = ARTMDatasetParquet(
        doc_index=doc_index,
        token_to_id=token_to_id,
        text_column=TEXT_COLUMN,
        tokenizer=custom_tokenizer,
        cache_dataframes=True,
    )

    # Шаг 4: DataLoader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        # collate_fn=make_artm_collate_fn(len(token_to_id)),
        collate_fn=ARTMCollator(len(token_to_id)),
        num_workers=4,
        shuffle=True,
    )

    # Тест: загрузить первый батч
    print("Testing first batch...")
    for doc_ids, bow in loader:
        print(f"doc_ids: {doc_ids}")
        print(f"Batch shape: {bow.shape}")
        print(f"Non-zero entries: {bow.nnz}")
        print(f"bow: {bow}")
        break

    print("Dataset is ready for ARTM training!")
