# artm_lib/artm/model_full_em_batched.py
from typing import Callable, Optional

import numpy as np
from scipy.sparse import csr_matrix


class FullEM_ARTM:
    def __init__(
        self,
        n_topics: int,
        vocab_size: int,
        reg_phi: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        reg_theta: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        random_state: Optional[int] = None,
        theta_convergence_tol: float = 1e-4,
        max_inner_iter: int = 20,
    ):
        """
        Векторизованная, батчевая реализация полного EM ARTM.
        """
        self.n_topics = n_topics
        self.vocab_size = vocab_size
        self.reg_phi = reg_phi
        self.reg_theta = reg_theta
        self.theta_convergence_tol = theta_convergence_tol
        self.max_inner_iter = max_inner_iter

        if random_state is not None:
            np.random.seed(random_state)

        # Инициализация Phi: (T, V)
        pw = np.ones(vocab_size) / vocab_size
        rand = np.random.rand(n_topics, vocab_size)
        self.phi = pw + rand
        self.phi /= self.phi.sum(axis=1, keepdims=True)

    def _normalize_rows(self, x: np.ndarray) -> np.ndarray:
        row_sums = x.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return x / row_sums

    def _normalize_cols(self, x: np.ndarray) -> np.ndarray:
        col_sums = x.sum(axis=0, keepdims=True)
        col_sums = np.where(col_sums == 0, 1.0, col_sums)
        return x / col_sums

    def _process_batch(self, bow: csr_matrix) -> tuple[np.ndarray, np.ndarray]:
        """
        Обрабатывает батч документов, возвращает:
        - n_wt_batch: (T, V) — накопленные n_{wt} для батча
        - theta_batch: (D, T) — финальные Theta для батча
        """
        D, V = bow.shape
        if V != self.vocab_size:
            raise ValueError("Vocab size mismatch")

        if D == 0:
            return np.zeros((self.n_topics, self.vocab_size)), np.zeros((0, self.n_topics))

        # Инициализация Theta для всего батча: (D, T)
        theta = np.full((D, self.n_topics), 1.0 / self.n_topics)

        # Предвычислим индексы и значения для всего батча
        # bow.data, bow.indices, bow.indptr — CSR формат
        doc_start = bow.indptr[:-1]
        doc_end = bow.indptr[1:]

        # Внутренний цикл по сходимости Theta (векторизован по батчу)
        for inner_iter in range(self.max_inner_iter):
            theta_old = theta.copy()

            # Сумма n_{tdw} по словам для каждого документа: (D, T)
            sum_ntdw = np.zeros_like(theta)

            # Накопление n_{tdw} для обновления n_{wt}
            # Мы будем агрегировать вклад каждого документа в n_wt
            # Но сначала — вычислим n_{tdw} для всех документов

            # Для каждого документа в батче
            for d in range(D):
                start, end = doc_start[d], doc_end[d]
                if start == end:  # пустой документ
                    continue

                w_indices = bow.indices[start:end]  # (nnz_d,)
                n_dw = bow.data[start:end]  # (nnz_d,)

                # phi_{wt} для слов документа d: (nnz_d, T)
                phi_wt = self.phi[:, w_indices].T  # (nnz_d, T)

                # theta_{td} для документа d: (1, T) → (nnz_d, T)
                theta_d = theta[d : d + 1]  # (1, T)

                # Шаг 7: n_{tdw} = n_{dw} * norm_t(phi_{wt} * theta_{td})
                unnorm_ntdw = phi_wt * theta_d  # (nnz_d, T)
                # Нормировка по t (axis=1)
                norm_factor = unnorm_ntdw.sum(axis=1, keepdims=True)
                norm_factor = np.where(norm_factor == 0, 1.0, norm_factor)
                ntdw = unnorm_ntdw / norm_factor  # (nnz_d, T)
                ntdw *= n_dw[:, None]  # (nnz_d, T)

                # Сумма по словам для theta update
                sum_ntdw[d] = ntdw.sum(axis=0)  # (T,)

                # !!! ВАЖНО: мы не можем сразу обновить n_wt,
                # потому что theta ещё не сошлась.
                # Поэтому обновление n_wt делаем ПОСЛЕ сходимости Theta.

            # Шаг 8: обновление Theta с регуляризацией
            if self.reg_theta is not None:
                grad_R_theta = self.reg_theta(theta)  # (D, T)
                theta_unnorm = sum_ntdw + theta * grad_R_theta
            else:
                theta_unnorm = sum_ntdw

            theta = self._normalize_rows(theta_unnorm)

            # Проверка сходимости (макс. изменение по всем документам)
            diff = np.abs(theta - theta_old).max()
            if diff < self.theta_convergence_tol:
                break

        # ТЕПЕРЬ, когда Theta сошлась, вычисляем финальные n_{tdw} и n_wt
        n_wt_batch = np.zeros((self.n_topics, self.vocab_size))

        for d in range(D):
            start = bow.indptr[d]
            end = bow.indptr[d + 1]
            if start == end:
                continue

            w_indices = bow.indices[start:end]
            n_dw = bow.data[start:end]

            phi_wt = self.phi[:, w_indices].T  # (nnz_d, T)
            theta_d = theta[d : d + 1]  # (1, T)

            unnorm_ntdw = phi_wt * theta_d
            norm_factor = unnorm_ntdw.sum(axis=1, keepdims=True)
            norm_factor = np.where(norm_factor == 0, 1.0, norm_factor)
            ntdw = unnorm_ntdw / norm_factor
            ntdw *= n_dw[:, None]

            # Агрегация в n_wt_batch
            for idx, w in enumerate(w_indices):
                n_wt_batch[:, w] += ntdw[idx, :]

        return n_wt_batch, theta

    def fit(self, data_loader, n_iter: int = 10) -> None:
        """
        Обучение по батчам.

        Parameters:
        -----------
        data_loader : torch.utils.data.DataLoader
            Возвращает (doc_ids, bow: csr_matrix)
        n_iter : int
            Число полных проходов по данным.
        """
        for epoch in range(n_iter):
            # Накопление n_wt по всему корпусу за эпоху
            n_wt_total = np.zeros((self.n_topics, self.vocab_size))

            for batch_idx, (doc_ids, bow) in enumerate(data_loader):
                n_wt_batch, _ = self._process_batch(bow)
                n_wt_total += n_wt_batch

            # Шаг 11: обновление Phi
            if self.reg_phi is not None:
                grad_R_phi = self.reg_phi(self.phi)
                phi_unnorm = n_wt_total + self.phi * grad_R_phi
            else:
                phi_unnorm = n_wt_total

            self.phi = self._normalize_cols(phi_unnorm)

            print(f"Epoch {epoch + 1}/{n_iter} completed")

    def get_phi(self) -> np.ndarray:
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
