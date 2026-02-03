# artm_lib/plsa/model.py
from typing import Optional, Any

import numpy as np
from scipy.sparse import csr_matrix


class PLSA:
    def __init__(self, n_topics: int, vocab_size: int, random_state: Optional[int] = None):
        """
        Реализация pLSA (probabilistic Latent Semantic Analysis).

        Parameters:
        -----------
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
        self.phi: np.ndarray = np.random.rand(n_topics, vocab_size).astype(np.float64)
        self.phi /= self.phi.sum(axis=1, keepdims=True)

    def _e_step(self, bow: csr_matrix) -> tuple[np.ndarray, np.ndarray]:
        """
        E-step: вычисление P(t | d, w) для всех слов в документах.

        Returns:
        --------
        n_tw : np.ndarray, shape (T, V)
            Ожидаемые подсчёты для обновления Phi.
        expected_counts : np.ndarray, shape (D, T)
            Ожидаемые подсчёты n_dw * P(t|d,w), суммированные по w.
        """
        D, V = bow.shape
        T = self.n_topics

        if bow.dtype != np.float64:
            bow = bow.astype(np.float64)

        # Начальное приближение theta
        unnorm_theta = bow @ self.phi.T  # (D, T)
        theta = unnorm_theta / np.maximum(unnorm_theta.sum(axis=1, keepdims=True), 1e-12)

        # Подготовка
        doc_start = bow.indptr[:-1]
        doc_end = bow.indptr[1:]
        expected_counts = np.zeros((D, T), dtype=np.float64)
        n_tw = np.zeros((T, V), dtype=np.float64)

        for d in range(D):
            start, end = doc_start[d], doc_end[d]
            if start == end:
                continue
            w_indices = bow.indices[start:end]
            n_dw = bow.data[start:end]

            theta_d = theta[d]
            phi_tw = self.phi[:, w_indices]

            # P(t | d, w) ∝ phi_tw * theta_d[:, None]
            unnorm_ptdw = phi_tw * theta_d[:, None]
            ptdw = unnorm_ptdw / np.maximum(unnorm_ptdw.sum(axis=0, keepdims=True), 1e-12)

            # Ожидаемые подсчёты
            exp_counts = ptdw * n_dw

            # Агрегация
            for idx, w in enumerate(w_indices):
                n_tw[:, w] += exp_counts[:, idx]

            expected_counts[d] = exp_counts.sum(axis=1)

        return n_tw, expected_counts

    def _m_step(self, n_tw: np.ndarray) -> None:
        """M-step: обновление Phi = P(w | t)"""
        self.phi = n_tw / np.maximum(n_tw.sum(axis=1, keepdims=True), 1e-12)

    def fit_batch(self, bow: csr_matrix) -> None:
        """Одна итерация EM на батче."""
        n_tw, _ = self._e_step(bow)
        self._m_step(n_tw)

    def fit(
        self,
        data_loader: Any,
        n_epochs: int = 10,
        val_loader: Optional[Any] = None,
        patience: Optional[int] = None,
        min_delta: float = 1.0,
        verbose: bool = True,
    ) -> dict:
        """
        Полное обучение по DataLoader.
        """
        history = {"train_perplexity": [], "val_perplexity": []}

        best_val_perp = float("inf")
        patience_counter = 0
        best_phi = None

        for epoch in range(n_epochs):
            # Обучение на всех батчах
            for doc_ids, bow in data_loader:
                self.fit_batch(bow)

            # Оценка на обучающей выборке (через тот же метод, что и в fit_full)
            train_perp = self.score_perplexity(data_loader)
            history["train_perplexity"].append(train_perp)

            if verbose:
                print(f"Epoch {epoch + 1}/{n_epochs}: train_perplexity={train_perp:.2f}")

            # Оценка на валидации
            if val_loader is not None:
                val_perp = self.score_perplexity(val_loader)
                history["val_perplexity"].append(val_perp)

                if verbose:
                    print(f"  val_perplexity={val_perp:.2f}")

                # Ранняя остановка
                if patience is not None:
                    if val_perp < best_val_perp - min_delta:
                        best_val_perp = val_perp
                        patience_counter = 0
                        best_phi = self.phi.copy()
                        if verbose:
                            print(f"  (new best, saved)")
                    else:
                        patience_counter += 1
                        if verbose:
                            print(f"  (no improvement, patience {patience_counter}/{patience})")

                    if patience_counter >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch + 1}")
                        if best_phi is not None:
                            self.phi = best_phi
                        break

        return history

    def transform(self, bow: csr_matrix) -> np.ndarray:
        """
        Получить Theta = P(t | d) для документов через E-step.
        """
        _, expected_counts = self._e_step(bow)
        theta = expected_counts / np.maximum(expected_counts.sum(axis=1, keepdims=True), 1e-12)
        return theta

    def get_phi(self) -> np.ndarray:
        return self.phi.copy()

    def get_top_words(self, vocab: list[str], top_n: int = 10) -> list[list[str]]:
        """Получить топ-слова для каждой темы."""
        top_words = []
        for t in range(self.n_topics):
            top_indices = np.argsort(self.phi[t])[::-1][:top_n]
            top_words.append([vocab[i] for i in top_indices])
        return top_words

    def _compute_log_likelihood(self, bow: csr_matrix, theta: np.ndarray) -> tuple[float, int]:
        """
        Вычисление лог-правдоподобия.
        """
        D, V = bow.shape

        if bow.dtype != np.float64:
            bow = bow.astype(np.float64)

        p_w_given_d = theta @ self.phi
        p_w_given_d = np.maximum(p_w_given_d, 1e-12)

        log_likelihood = np.float64(0.0)
        total_words = 0

        doc_start = bow.indptr[:-1]
        doc_end = bow.indptr[1:]

        for d in range(D):
            start, end = doc_start[d], doc_end[d]
            if start == end:
                continue

            w_indices = bow.indices[start:end]
            n_dw = bow.data[start:end]
            p_wd = p_w_given_d[d, w_indices]

            log_likelihood += np.sum(n_dw * np.log(p_wd))
            total_words += int(np.sum(n_dw))

        return float(log_likelihood), total_words

    def score_perplexity(self, data_loader: Any) -> float:
        """
        Вычисляет перплексию на данных из DataLoader.
        ИСПОЛЬЗУЕТ transform() — единообразно с логикой оценки.
        """
        total_log_likelihood = np.float64(0.0)
        total_words = 0

        for doc_ids, bow in data_loader:
            if bow.shape[0] == 0:
                continue

            # Используем transform() — тот же метод, что используется вне обучения
            theta = self.transform(bow)
            log_likelihood, words = self._compute_log_likelihood(bow, theta)
            total_log_likelihood += log_likelihood
            total_words += words

        if total_words == 0:
            return float("inf")

        return np.exp(-total_log_likelihood / total_words)

    def score_perplexity_from_matrix(self, X: csr_matrix) -> float:
        """Перплексия для полной матрицы (без DataLoader)."""
        theta = self.transform(X)
        log_likelihood, total_words = self._compute_log_likelihood(X, theta)
        return np.exp(-log_likelihood / total_words) if total_words > 0 else float("inf")

    def fit_full(
        self,
        X: csr_matrix,
        max_iter: int = 100,
        tol: float = 1e-4,
        verbose: bool = True,
    ) -> dict:
        """
        Классический EM на полной матрице без батчей.

        ИСПРАВЛЕНО: Теперь использует transform() для согласованности с fit().
        """
        history = {"train_perplexity": []}

        for iter in range(max_iter):
            # E-step с получением expected_counts
            n_tw, expected_counts = self._e_step_full(X)

            # M-step
            old_phi = self.phi.copy()
            self._m_step(n_tw)

            diff = np.linalg.norm(self.phi - old_phi)

            # ИСПРАВЛЕНО: Используем transform() вместо прямого вычисления из expected_counts
            # Это гарантирует единообразие с score_perplexity()
            theta = self.transform(X)
            log_lik, n_words = self._compute_log_likelihood(X, theta)
            perp = np.exp(-log_lik / n_words) if n_words > 0 else float("inf")
            history["train_perplexity"].append(perp)

            if verbose and iter % 10 == 0:
                print(f"Iter {iter}: perplexity={perp:.2f}, diff={diff:.6f}")

            if diff < tol:
                if verbose:
                    print(f"Converged at iteration {iter}")
                break

        return history

    def _e_step_full(self, X: csr_matrix) -> tuple[np.ndarray, np.ndarray]:
        """Оптимизированный E-step для полной матрицы."""
        D, V = X.shape
        T = self.n_topics

        if X.dtype != np.float64:
            X = X.astype(np.float64)

        unnorm_theta = X @ self.phi.T
        theta = unnorm_theta / np.maximum(unnorm_theta.sum(axis=1, keepdims=True), 1e-12)

        n_tw = np.zeros((T, V), dtype=np.float64)
        expected_counts = np.zeros((D, T), dtype=np.float64)

        for d in range(D):
            start, end = X.indptr[d : d + 2]
            if start == end:
                continue
            words = X.indices[start:end]
            counts = X.data[start:end]
            theta_d = theta[d]

            phi_w = self.phi[:, words]
            unnorm_ptdw = phi_w * theta_d[:, None]
            ptdw = unnorm_ptdw / np.maximum(unnorm_ptdw.sum(axis=0, keepdims=True), 1e-12)

            exp_counts = ptdw * counts
            n_tw[:, words] += exp_counts
            expected_counts[d] = exp_counts.sum(axis=1)

        return n_tw, expected_counts
