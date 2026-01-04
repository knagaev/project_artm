# Только коллаторы — ничего лишнего
from scipy.sparse import coo_matrix
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
