"""# ARTM-реализация
from scipy.sparse import csr_matrix


class ARTM:
    def __init__(self, n_topics, vocab_size):
        pass

    def fit_batch(self, bow_matrix: csr_matrix):
        pass

    def transform(self, bow_matrix):
        pass
"""

# artm_lib/artm/model.py
from typing import Callable, Optional

import numpy as np
from scipy.sparse import csr_matrix


class ARTM:
    def __init__(
        self,
        n_topics: int,
        vocab_size: int,
        phi: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        theta: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        random_state: Optional[int] = None,
        theta_convergence_tol: float = 1e-4,
        max_inner_iter: int = 50,
        eps: float = 1e-12,
    ):
        """
        Полная реализация ARTM по алгоритму из описания:
        - Внешний цикл по итерациям (imax)
        - Внутренний цикл по сходимости Theta для каждого документа
        - Поддержка произвольных регуляризаторов через градиенты

        Parameters:
        -----------
        n_topics : int
            Число тем T.
        vocab_size : int
            Размер словаря V.
        phi : callable or None
            Функция grad_R_phi(phi) -> gradient of R w.r.t. phi (shape: T x V)
        theta : callable or None
            Функция grad_R_theta(theta) -> gradient of R w.r.t. theta (shape: D x T)
            Но так как Theta обновляется по одному документу, принимает (1, T)
        random_state : int, optional
        theta_convergence_tol : float
            Порог сходимости для внутреннего цикла по Theta.
        max_inner_iter : int
            Макс. число итераций для сходимости Theta.
        """
        self.n_topics = n_topics
        self.vocab_size = vocab_size
        self.phi = phi
        self.theta = theta
        self.theta_convergence_tol = theta_convergence_tol
        self.max_inner_iter = max_inner_iter
        self._eps: float = eps

        if random_state is not None:
            np.random.seed(random_state)

        # Initialize Phi: T x V, rows sum to 1
        # Инициализация Phi: W x T (в алгоритме ϕ_{wt}, w ∈ W, t ∈ T)
        # Но удобнее хранить как T x V → phi[t, w]
        pw = np.ones(vocab_size) / vocab_size  # равномерное распределение слов
        rand = np.random.rand(n_topics, vocab_size)
        self.phi = pw + rand  # broadcasting: (V,) + (T, V) → (T, V)
        # Нормировка по w ∈ W для каждой темы t (строки = темы)
        self.phi /= self.phi.sum(axis=1, keepdims=True)  # (T, V)

    def _normalize_rows(self, x: np.ndarray) -> np.ndarray:
        """Нормировка каждой строки до суммы 1."""
        row_sums = x.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return x / row_sums

    def _normalize_cols(self, x: np.ndarray) -> np.ndarray:
        """Нормировка каждого столбца до суммы 1 (для Phi по w)."""
        col_sums = x.sum(axis=0, keepdims=True)
        col_sums = np.where(col_sums == 0, 1.0, col_sums)
        return x / col_sums

    def _norm(self, x: np.ndarray) -> np.ndarray:
        # take x+ = max(x, 0) element-wise (perform projection on positive simplex)
        x = np.maximum(x, np.zeros_like(x))
        # normalize values in non-zero rows to 1
        # (mapping from the positive simplex to the unit simplex)
        norm = x.sum(axis=0)
        x = np.where(norm > self._eps, x / norm, np.zeros_like(x))
        return x

    '''    
    def _e_step(self, bow: csr_matrix) -> np.ndarray:
        """
        Simplified E-step for offline ARTM:
        Compute Theta ~ alpha + bow @ Phi^T.
        This avoids explicit P(t|d,w) and is standard in fast offline ARTM.
        """
        # Compute unnormalized Theta: (D, T)
        unnorm_theta = bow @ self.phi.T  # (D, T)
        unnorm_theta += self.alpha

        # Normalize safely
        row_sums = unnorm_theta.sum(axis=1, keepdims=True)
        # Avoid division by zero (empty documents)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        theta = unnorm_theta / row_sums
        return theta

    def _m_step(self, bow: csr_matrix, theta: np.ndarray) -> None:
        """
        M-step: update Phi matrix.

        Parameters:
        -----------
        bow : scipy.sparse.csr_matrix of shape (D, V)
            Bag-of-words matrix.
        theta : np.ndarray of shape (D, T)
            Document-topic distributions from E-step.
        """
        D, V = bow.shape
        T = self.n_topics

        # Compute expected counts: n_tw = sum_d n_dw * P(t | d)
        # Result: (T, V)
        # We use: n_tw = theta.T @ bow
        n_tw = theta.T @ bow  # (T, V)

        # Apply regularizers:
        # - Smoothing (beta): acts as prior count
        # - Sparsity (sparsity_phi): subtracts constant (smoothed version)
        regularized_n_tw = n_tw + self.beta

        if self.sparsity_phi > 0:
            # Smoothed sparsity: subtract average per topic
            # This encourages many words to have near-zero prob
            avg_per_topic = regularized_n_tw.mean(axis=1, keepdims=True)
            regularized_n_tw -= self.sparsity_phi * avg_per_topic

        # Ensure non-negative
        regularized_n_tw = np.maximum(regularized_n_tw, 1e-12)

        # Normalize to get new Phi
        self.phi = regularized_n_tw / regularized_n_tw.sum(axis=1, keepdims=True)

    def fit_batch(self, bow: csr_matrix) -> None:
        """
        Perform one EM iteration on a batch of documents.

        Parameters:
        -----------
        bow : scipy.sparse.csr_matrix of shape (D, V)
        """
        if bow.shape[1] != self.vocab_size:
            raise ValueError(f"BoW vocab size {bow.shape[1]} != model vocab {self.vocab_size}")

        # E-step
        theta = self._e_step(bow)

        # M-step
        self._m_step(bow, theta)

        # Store theta for last batch (optional)
        self.theta = theta

    def fit(self, data_loader, n_epochs: int = 10) -> None:
        """
        Full offline training: iterate over entire dataset for n_epochs.

        Parameters:
        -----------
        data_loader : torch.utils.data.DataLoader
            Must yield (doc_ids, bow_matrix) where bow_matrix is csr_matrix.
        n_epochs : int
            Number of passes over the data.
        """
        for epoch in range(n_epochs):
            for batch_idx, (doc_ids, bow) in enumerate(data_loader):
                self.fit_batch(bow)
            print(f"Epoch {epoch + 1}/{n_epochs} completed")
    '''

    def get_phi(self) -> np.ndarray:
        """Return the Phi matrix (T x V)."""
        return self.phi.copy()

    def get_top_words(self, vocab: list[str], top_n: int = 10) -> list[list[str]]:
        """
        Get top words for each topic.

        Parameters:
        -----------
        vocab : list[str]
            Vocabulary list of size V, where vocab[i] = word for token_id=i.
        top_n : int
            Number of top words to return per topic.

        Returns:
        --------
        topics : list[list[str]]
            Top words for each topic.
        """
        if len(vocab) != self.vocab_size:
            raise ValueError("Vocab size mismatch")
        top_words = []
        for t in range(self.n_topics):
            top_indices = np.argsort(self.phi[t])[::-1][:top_n]
            top_words.append([vocab[i] for i in top_indices])
        return top_words

    def fit(self, bow_full: csr_matrix, n_iter: int = 10) -> None:
        """
        Обучение ARTM по полному корпусу.

        Parameters:
        -----------
        bow_full : csr_matrix of shape (D, V)
            Весь корпус в виде BoW.
        n_iter : int
            Число внешних итераций (imax).
        """
        D, V = bow_full.shape
        if V != self.vocab_size:
            raise ValueError(f"Vocab size mismatch: {V} vs {self.vocab_size}")

        # Внешний цикл
        for i in range(n_iter):
            # n_{wt} = 0 для всех w, t → shape (T, V)
            n_wt = np.zeros((self.n_topics, self.vocab_size), dtype=np.float64)

            # Проход по всем документам
            for d in range(D):
                # Извлекаем BoW для документа d
                bow_d = bow_full[d]  # (1, V)
                if bow_d.nnz == 0:
                    continue

                # Индексы ненулевых слов
                w_indices = bow_d.indices  # array of word ids
                n_dw = bow_d.data  # array of counts, shape (nnz,)

                # Инициализация Theta: theta_t = 1/|T| для всех t
                theta = np.full((1, self.n_topics), 1.0 / self.n_topics)  # (1, T)

                # Внутренний цикл: пока Theta не сойдётся
                for inner_iter in range(self.max_inner_iter):
                    theta_old = theta.copy()

                    # Шаг 7: n_{tdw} = n_{dw} * norm_{t∈T}(phi_{wt} * theta_{td})
                    # phi_wt для w ∈ d: shape (nnz, T)
                    phi_wt = self.phi[:, w_indices].T  # (nnz, T)
                    # theta_td: (1, T) → broadcast to (nnz, T)
                    unnorm_ntdw = phi_wt * theta  # (nnz, T)
                    # Нормировка по t ∈ T для каждого w
                    ntdw = unnorm_ntdw / unnorm_ntdw.sum(axis=1, keepdims=True)  # (nnz, T)
                    # Умножение на n_dw
                    ntdw *= n_dw[:, None]  # (nnz, T)

                    # Шаг 8: theta_td = norm_{t∈T}( sum_{w∈d} ntdw + theta_td * dR/dtheta_td )
                    sum_ntdw = ntdw.sum(axis=0, keepdims=True)  # (1, T)

                    if self.theta is not None:
                        grad_R_theta = self.theta(theta)  # (1, T)
                        theta_unnorm = sum_ntdw + theta * grad_R_theta
                    else:
                        theta_unnorm = sum_ntdw

                    theta = self._normalize_rows(theta_unnorm)

                    # Проверка сходимости
                    diff = np.abs(theta - theta_old).max()
                    if diff < self.theta_convergence_tol:
                        break

                # Шаг 10: n_{wt} += n_{tdw} для всех w ∈ d, t ∈ T
                # ntdw: (nnz, T), w_indices: (nnz,)
                for idx, w in enumerate(w_indices):
                    n_wt[:, w] += ntdw[idx, :]  # (T,) += (T,)

            # Шаг 11: phi_{wt} = norm_{w∈W}( n_{wt} + phi_{wt} * dR/dphi_{wt} )
            if self.phi is not None:
                grad_R_phi = self.phi(self.phi)  # (T, V)
                phi_unnorm = n_wt + self.phi * grad_R_phi
            else:
                phi_unnorm = n_wt

            # Нормировка по w ∈ W (столбцы) для каждой темы t
            self.phi = self._normalize_cols(phi_unnorm)

            print(f"Iteration {i + 1}/{n_iter} completed")
