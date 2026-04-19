import numpy as np

def attn_3d(X, indices, gamma, beta=0.5):
    X = np.asarray(X, dtype=float)
    indices = np.asarray(indices, dtype=int)
    K, I = X.shape
    
    if gamma == 1.0:
        return X.copy()
    
    alpha = 1.0 - gamma
    arange = np.arange(I)
    alpha_pow = alpha ** arange
    alpha_inv_pow = 1.0 / alpha_pow

    def calc_right_ema(X_in, idx_in):
        W = X_in * alpha_inv_pow
        global_cum = np.cumsum(W, axis=1)
        
        resets = idx_in[1:-1]
        mask = np.zeros(I, dtype=int)
        if len(resets) > 0:
            mask[resets] = resets
            
        last_reset = np.maximum.accumulate(mask)
        correction = np.zeros((K, I))
        valid = last_reset > 0
        if np.any(valid):
            correction[:, valid] = global_cum[:, last_reset[valid] - 1]
        
        seg_cum = global_cum - correction
        
        lengths = np.diff(idx_in)
        starts = idx_in[:-1]
        X_starts = np.repeat(X_in[:, starts], lengths, axis=1)
        alpha_inv_starts = np.repeat(alpha_inv_pow[starts], lengths)
        
        term1 = alpha_pow * alpha * alpha_inv_starts * X_starts
        term2 = gamma * alpha_pow * seg_cum
        return term1 + term2

    Y = calc_right_ema(X, indices)
    
    X_rev = X[:, ::-1]
    rev_indices = I - indices[::-1]
    Z_rev = calc_right_ema(X_rev, rev_indices)
    Z = Z_rev[:, ::-1]
    
    return beta * Y + (1.0 - beta) * Z

n = 5000
X = np.arange(n) + 1;
Y = np.arange(n) + 2;
Z = np.vstack((X, Y));
indices = [0, n];
gamma = 0.1;
beta = 0.5

print(attn_3d(Z, indices, gamma))