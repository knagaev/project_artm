# plsa_numpy.py
import numpy as np

# import torch
# from scipy.special import logsumexp
# from torch.utils.data import DataLoader, Dataset


# ---------- 1. Dataset, который возвращает (doc_id, word_id, count) ----------
class BagOfWordsDataset(Dataset):
    """
    Принимает список документов в виде списка списков токен-идов
    [[w1,w2,...], ...] и строит разреженную матрицу N_{dw}.
    Возвращает тройки (d, w, n_dw) для всех ненулевых элементов.
    """

    def __init__(self, tokenised_docs, vocab_size):
        self.vocab_size = vocab_size
        from collections import Counter

        rows, cols, vals = [], [], []
        for d, doc in enumerate(tokenised_docs):
            cnt = Counter(doc)
            for w, c in cnt.items():
                rows.append(d)
                cols.append(w)
                vals.append(float(c))
        self.rows = np.array(rows, dtype=np.int32)
        self.cols = np.array(cols, dtype=np.int32)
        self.vals = np.array(vals, dtype=np.float32)
        self.n_docs = len(tokenised_docs)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return (self.rows[idx], self.cols[idx], self.vals[idx])


# ---------- 2. Собственно PLSA ----------
class PLSA:
    def __init__(self, n_topics, max_iter=100, tol=1e-4, smooth=1e-8):
        self.K = n_topics
        self.max_iter = max_iter
        self.tol = tol
        self.eps = smooth

    def fit(self, loader: DataLoader, n_docs: int, vocab_size: int, verbose=True):
        """
        loader должен выдавать тензоры shape=(3,)  [d, w, n_dw].
        n_docs и vocab_size нужны для инициализации матриц.
        """
        # Собираем всё в плотные numpy-матрицы
        N_dw = np.zeros((n_docs, vocab_size), dtype=np.float32)
        for d, w, c in loader:
            N_dw[d.long(), w.long()] = c.float()
        N_d = N_dw.sum(axis=1, keepdims=True)  # (D,1)

        # Инициализация θ и φ (нормализуем случайные числа)
        self.theta = np.random.dirichlet(np.ones(self.K), size=n_docs).T
        self.phi = np.random.dirichlet(np.ones(vocab_size), size=self.K)

        ll_old = -np.inf
        for it in range(self.max_iter):
            # E-step: P(z|d,w)  (shape K,D,W)
            tmp = self.theta[:, :, None] * self.phi[:, None, :]
            P_z_dw = tmp / (tmp.sum(axis=0, keepdims=True) + self.eps)

            # M-step
            # n_dw  (D,W)  →  (1,D,W) для вещания
            # P(z|d,w)*n_dw  →  (K,D,W)
            resp = P_z_dw * N_dw[None, :, :]

            # θ_dk ∝ sum_w  resp * n_dw
            self.theta = resp.sum(axis=2)  # (K,D)
            self.theta /= self.theta.sum(axis=0, keepdims=True) + self.eps

            # φ_kw ∝ sum_d  resp * n_dw
            self.phi = resp.sum(axis=1)  # (K,W)
            self.phi /= self.phi.sum(axis=1, keepdims=True) + self.eps

            # лог-правдоподобие
            tmp = self.theta.T @ self.phi  # (D,W)
            ll = (N_dw * np.log(tmp + self.eps)).sum()
            if verbose:
                print(f"iter {it + 1:3d}  logL={ll:,.0f}  Δ={ll - ll_old:+,.2f}")
            if abs(ll - ll_old) < self.tol:
                break
            ll_old = ll
        return self

    # ---------- 3. Инференс для нового документа ----------
    def transform(self, bow, max_iter=20):
        """
        bow – вектор длины vocab_size с частотами слов.
        Возвращает θ_d  (K,)
        """
        n = bow.sum()
        theta_d = np.ones(self.K) / self.K
        for _ in range(max_iter):
            # E: P(z|w) ∝ φ_kw * θ_k
            P_z_w = self.phi * theta_d[:, None]  # (K,V)
            P_z_w /= P_z_w.sum(axis=0, keepdims=True) + self.eps
            # M: θ_k ∝ sum_w  P(z|w) * n_w
            theta_d = (P_z_w @ bow) + self.eps
            theta_d /= theta_d.sum()
        return theta_d


# ---------- 4. Пример использования ----------
if __name__ == "__main__":
    # псевдокорпус из 500 документов, словарь 1000 токенов
    docs = [np.random.randint(0, 1000, size=np.random.poisson(80)) for _ in range(500)]
    vocab_size = 1000
    dataset = BagOfWordsDataset(docs, vocab_size)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    model = PLSA(n_topics=20, max_iter=50).fit(loader, n_docs=len(docs), vocab_size=vocab_size)

    # топ-слова темы 0
    topw = 10
    for k in range(min(3, model.K)):
        idx = np.argpartition(-model.phi[k], topw)[:topw]
        print(f"topic {k}:", idx)
