from __future__ import annotations

import numpy as np
import jax
from jax import Array

from .metric_base import Metric


class PerplexityMetric(Metric):
    def __init__(
        self,
        tag: str | None = None,
        eps: float = 1e-12,
        chunk_size: int = 10000,
    ):
        """
        Memory-safe perplexity computation on host (NumPy), chunk by chunk.

        Computes exactly the same metric:
            p(w_i | C_i) = sum_t phi_it[i, t] * theta[i, t]
            L = sum_i log p(w_i | C_i)
            perplexity = exp(-L / I)

        but avoids building a large JAX device computation graph.
        """
        if tag is None:
            tag = self.__class__.__name__
        super().__init__(tag=tag)

        self._eps = float(eps)
        self._chunk_size = int(chunk_size)

    def _call_impl(self, phi_it: Array, phi_wt: Array, theta: Array) -> float:
        if phi_it.shape != theta.shape:
            raise ValueError(
                f"PerplexityMetric expects phi_it and theta to have same shape, "
                f"got phi_it={phi_it.shape}, theta={theta.shape}"
            )

        num_words = int(phi_it.shape[0])
        if num_words == 0:
            return float("nan")

        total_log_likelihood = 0.0

        for start in range(0, num_words, self._chunk_size):
            end = min(start + self._chunk_size, num_words)

            # Transfer only a small chunk to host
            phi_chunk = np.asarray(jax.device_get(phi_it[start:end]), dtype=np.float32)
            theta_chunk = np.asarray(jax.device_get(theta[start:end]), dtype=np.float32)

            # p(w_i | C_i) = sum_t p(w_i|t) p(t|C_i)
            p_wi_chunk = np.sum(phi_chunk * theta_chunk, axis=1)

            total_log_likelihood += float(np.sum(np.log(p_wi_chunk + self._eps)))

            # Explicitly drop chunk buffers as soon as possible
            del phi_chunk
            del theta_chunk
            del p_wi_chunk

        perplexity = np.exp(-total_log_likelihood / num_words)
        return float(perplexity)