from typing import Callable

import polars as pl

# from torch.utils.data import Dataset


class ARTMDatasetParquet:
    def __init__(
        self,
        doc_index: list[tuple[str, int]],
        token_to_id: dict,
        text_column: str,
        tokenizer: Callable[[str], list[str]],
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
                self._df_cache[file_path] = pl.read_parquet(file_path, columns=[self.text_column])
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
