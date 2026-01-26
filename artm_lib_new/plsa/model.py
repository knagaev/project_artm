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

    def score_perplexity_from_matrix(self, X: csr_matrix) -> float:
        """Более эффективная перплексия для полной матрицы."""
        theta = self.transform(X)
        p_w_given_d = theta @ self.phi
        p_w_given_d = np.maximum(p_w_given_d, 1e-12)

        # Лог-правдоподобие через sparse
        log_likelihood = 0.0
        total_words = 0

        for d in range(X.shape[0]):
            start, end = X.indptr[d : d + 2]
            if start == end:
                continue
            words = X.indices[start:end]
            counts = X.data[start:end]
            p_wd = p_w_given_d[d, words]
            log_likelihood += np.sum(counts * np.log(p_wd))
            total_words += np.sum(counts)

        return np.exp(-log_likelihood / total_words) if total_words > 0 else float("inf")

    def fit_full(self, X: csr_matrix, max_iter: int = 100, tol: float = 1e-4):
        """Классический EM на полной матрице."""
        for iter in range(max_iter):
            # E-step: вычисляем n_tw для всей матрицы
            n_tw, theta = self._e_step_full(X)

            # Сохраняем theta для последующего использования
            self.theta = theta

            # M-step: обновляем phi
            old_phi = self.phi.copy()
            self._m_step(n_tw)

            # Проверка сходимости
            if np.linalg.norm(self.phi - old_phi) < tol:
                print(f"Сошлось на итерации {iter}")
                break

    def _e_step_full(self, X: csr_matrix):
        """Оптимизированный E-step для полной матрицы."""
        D, V = X.shape
        T = self.n_topics

        # Вычисляем theta = X @ phi.T (нормированный)
        unnorm_theta = X @ self.phi.T
        theta = unnorm_theta / np.maximum(unnorm_theta.sum(axis=1, keepdims=True), 1e-12)

        # Вычисляем n_tw = phi.T @ (X.multiply(theta_sum))
        # Более эффективный способ:
        n_tw = np.zeros((T, V))

        # Используем sparse operations
        for d in range(D):
            start, end = X.indptr[d : d + 2]
            if start == end:
                continue
            words = X.indices[start:end]
            counts = X.data[start:end]
            theta_d = theta[d]  # (T,)

            # Векторизованное вычисление
            phi_w = self.phi[:, words]  # (T, nnz)
            denom = np.dot(theta_d, phi_w)  # (nnz,)
            denom = np.maximum(denom, 1e-12)

            # n_tdw = counts * (phi_w * theta_d[:, None]) / denom
            n_tdw = counts[None, :] * phi_w * theta_d[:, None] / denom[None, :]
            n_tw[:, words] += n_tdw

        return n_tw, theta
