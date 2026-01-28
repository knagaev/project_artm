# collators.py
# Только коллаторы — ничего лишнего
import numpy as np
from scipy.sparse import csr_matrix


class ARTMCollator:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def __call__(self, batch: list[tuple[int, list[int]]]) -> tuple[list[int], csr_matrix]:
        """
        Преобразует батч документов в разреженную BOW-матрицу с частотами.

        Args:
            batch: list[(doc_id, list[token_id])]

        Returns:
            doc_ids: list[int] — идентификаторы документов в батче
            bow_matrix: csr_matrix of shape (len(batch), vocab_size) — частотная матрица
        """
        doc_ids = [item[0] for item in batch]
        list_of_token_ids = [item[1] for item in batch]

        # Если батч пуст или все документы пустые
        if not list_of_token_ids or all(len(x) == 0 for x in list_of_token_ids):
            bow_matrix = csr_matrix((len(batch), self.vocab_size), dtype=np.int32)
            return doc_ids, bow_matrix

        # Собираем координаты для разреженной матрицы
        all_tokens = []
        doc_indices = []
        for i, token_ids in enumerate(list_of_token_ids):
            if token_ids:
                all_tokens.extend(token_ids)
                doc_indices.extend([i] * len(token_ids))

        # Если после фильтрации нет токенов
        if not all_tokens:
            bow_matrix = csr_matrix((len(batch), self.vocab_size), dtype=np.int32)
            return doc_ids, bow_matrix

        # Преобразуем в массивы numpy
        all_tokens = np.array(all_tokens, dtype=np.int32)
        doc_indices = np.array(doc_indices, dtype=np.int32)

        # Фильтрация токенов вне допустимого диапазона [0, vocab_size)
        mask = (all_tokens >= 0) & (all_tokens < self.vocab_size)
        all_tokens = all_tokens[mask]
        doc_indices = doc_indices[mask]

        # Создаём разреженную матрицу
        # Важно: data = 1 для каждого вхождения → csr_matrix автоматически суммирует дубликаты!
        data = np.ones_like(all_tokens, dtype=np.int32)
        bow_matrix = csr_matrix(
            (data, (doc_indices, all_tokens)), shape=(len(batch), self.vocab_size), dtype=np.int32
        )

        return doc_ids, bow_matrix
