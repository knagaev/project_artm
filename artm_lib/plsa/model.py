# artm_lib/plsa/model.py
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix


class PLSA:
    def __init__(self, n_topics: int, vocab_size: int, random_state: Optional[int] = None):
        """
        Реализация pLSA (probabilistic Latent Semantic Analysis).

        Параметры:
        ----------
        n_topics : int
            Число тем T.
        vocab_size : int
            Размер словаря V.
        random_state : int, optional
            Для воспроизводимости.
        """
        self.n_topics: int = n_topics
        self.vocab_size: int = vocab_size

        if random_state is not None:
            np.random.seed(random_state)

        # Инициализация Phi: P(w | t) — shape (T, V)
        self.phi: np.ndarray = np.random.rand(n_topics, vocab_size)
        self.phi /= self.phi.sum(axis=1, keepdims=True)

        # Theta будет вычисляться на лету (не храним глобально)

    def _e_step(self, bow: csr_matrix) -> tuple[np.ndarray, np.ndarray]:
        """
        E-step: вычисление P(t | d, w) для всех слов в документах.

        Но так как мы не храним P(t | d, w) явно, вместо этого
        вычисляем матрицу ожидаемых подсчётов: n_{dw} * P(t | d, w)
        """
        D: int
        V: int
        D, V = bow.shape
        T: int = self.n_topics

        # Вычисляем unnorm P(t | d) ~ sum_w n_dw * phi_tw
        # Это приближение, но для M-step достаточно
        unnorm_theta: np.ndarray = bow @ self.phi.T  # (D, T)
        # Нормировка
        theta = unnorm_theta / np.maximum(unnorm_theta.sum(axis=1, keepdims=True), 1e-12)

        # Теперь вычисляем P(t | d, w) ∝ phi_tw * theta_dt
        # Но нам нужно только: n_{dw} * P(t | d, w)
        # Для этого используем sparse operations

        # Подготовка
        doc_start = bow.indptr[:-1]
        doc_end = bow.indptr[1:]
        expected_counts: np.ndarray = np.zeros((D, T))  # для Theta (опционально)
        n_tw: np.ndarray = np.zeros((T, V))  # для обновления Phi

        for d in range(D):
            start, end = doc_start[d], doc_end[d]
            if start == end:
                continue
            w_indices = bow.indices[start:end]
            n_dw = bow.data[start:end]

            # theta_dt для документа d: (T,)
            theta_d = theta[d]  # (T,)

            # phi_tw для слов w: (T, nnz)
            phi_tw = self.phi[:, w_indices]  # (T, nnz)

            # P(t | d, w) ∝ phi_tw * theta_d[:, None] → (T, nnz)
            unnorm_ptdw = phi_tw * theta_d[:, None]
            # Нормировка по t
            ptdw = unnorm_ptdw / np.maximum(unnorm_ptdw.sum(axis=0, keepdims=True), 1e-12)

            # Ожидаемые подсчёты: n_dw * P(t | d, w) → (T, nnz)
            exp_counts = ptdw * n_dw  # broadcasting: (T, nnz)

            # Агрегация в n_tw
            for idx, w in enumerate(w_indices):
                n_tw[:, w] += exp_counts[:, idx]

            # (Опционально) агрегация в expected_counts
            expected_counts[d] = exp_counts.sum(axis=1)

        return n_tw, expected_counts

    def _m_step(self, n_tw: np.ndarray) -> None:
        """M-step: обновление Phi = P(w | t)"""
        # Нормировка по словам для каждой темы
        self.phi = n_tw / np.maximum(n_tw.sum(axis=1, keepdims=True), 1e-12)

    def fit_batch(self, bow: csr_matrix) -> None:
        """Одна итерация EM на батче."""
        n_tw, _ = self._e_step(bow)
        self._m_step(n_tw)

    def fit(self, data_loader, n_epochs: int = 10) -> None:
        """Полное обучение по DataLoader."""
        for epoch in range(n_epochs):
            for batch_idx, (doc_ids, bow) in enumerate(data_loader):
                self.fit_batch(bow)
            print(f"Epoch {epoch + 1}/{n_epochs} completed")

    def transform(self, bow: csr_matrix) -> np.ndarray:
        """Получить Theta = P(t | d) для новых документов."""
        unnorm_theta = bow @ self.phi.T
        theta = unnorm_theta / np.maximum(unnorm_theta.sum(axis=1, keepdims=True), 1e-12)
        return theta

    def get_phi(self) -> np.ndarray:
        return self.phi.copy()

    def get_top_words(self, vocab: list, top_n: int = 10) -> list:
        """Получить топ-слова для каждой темы."""
        top_words = []
        for t in range(self.n_topics):
            top_indices = np.argsort(self.phi[t])[::-1][:top_n]
            top_words.append([vocab[i] for i in top_indices])
        return top_words

    # ... предыдущий код PLSA ...

    def score_perplexity(self, data_loader) -> float:
        """
        Вычисляет перплексию модели на данных из DataLoader.

        Parameters:
        -----------
        data_loader : torch.utils.data.DataLoader
            Должен возвращать (doc_ids, bow: csr_matrix)

        Returns:
        --------
        perplexity : float
        """
        total_log_likelihood = 0.0
        total_words = 0

        for doc_ids, bow in data_loader:
            # bow: (D, V)
            D, V = bow.shape

            if D == 0:
                continue

            # Шаг 1: вычислить P(t | d) для всех документов в батче
            unnorm_theta = bow @ self.phi.T  # (D, T)
            theta = unnorm_theta / np.maximum(unnorm_theta.sum(axis=1, keepdims=True), 1e-12)

            # Шаг 2: вычислить P(w | d) = sum_t P(w | t) * P(t | d)
            # Результат: (D, V)
            p_w_given_d = theta @ self.phi  # (D, T) @ (T, V) -> (D, V)

            # Защита от log(0)
            p_w_given_d = np.maximum(p_w_given_d, 1e-12)

            # Шаг 3: вычислить лог-правдоподобие: sum_{d,w} n_dw * log P(w|d)
            # Используем sparse: только ненулевые элементы
            doc_start = bow.indptr[:-1]
            doc_end = bow.indptr[1:]

            for d in range(D):
                start, end = doc_start[d], doc_end[d]
                if start == end:
                    continue

                w_indices = bow.indices[start:end]
                n_dw = bow.data[start:end]

                # P(w | d) для этих слов
                p_wd = p_w_given_d[d, w_indices]  # (nnz,)

                # Лог-правдоподобие для документа d
                log_likelihood_d = np.sum(n_dw * np.log(p_wd))
                total_log_likelihood += log_likelihood_d
                total_words += np.sum(n_dw)

        if total_words == 0:
            return float("inf")

        perplexity = np.exp(-total_log_likelihood / total_words)
        return perplexity
