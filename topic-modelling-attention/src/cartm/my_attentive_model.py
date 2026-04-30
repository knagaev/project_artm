"""
Реализация Attentive ARTM на JAX.

Модуль повторяет EM-схему из статьи: на E-шаге уточняются локальные
распределения тем по позициям токенов с учетом контекста, на M-шаге
пересчитывается матрица ``phi`` с поправкой на локальное внимание и
регуляризаторы.
"""
#from rich.panel import p

from functools import partial
from typing import Callable

import numpy as np
from numpy.typing import NDArray
import jax
import jax.numpy as jnp
from jax import Array

from . import metrics as mtc
from . import regularization as reg

@jax.jit
def _norm_jax(x: jax.Array, eps: float = 1e-10) -> jax.Array:
    x = jnp.maximum(x, jnp.zeros_like(x))
    norm = x.sum(axis=0)
    return jnp.where(norm > eps, x / norm, jnp.zeros_like(x))

def _norm_numpy(x: NDArray, eps: float = 1e-6) -> NDArray:
    x = np.maximum(x, np.zeros_like(x))
    norm = x.sum(axis=0)
    return np.divide(x, norm, out=np.zeros_like(x), where=norm != 0)

@jax.jit
def _process_row(x: Array, indices: Array, gamma: float, beta: float) -> Array:
    alpha = 1.0 - gamma
    seq_len = x.shape[0]
    
    # Маска сброса для прямого прохода (начало сегмента)
    is_start = jnp.zeros(seq_len, dtype=bool).at[indices[:-1]].set(True)
    # Маска сброса для обратного прохода (конец сегмента)
    is_end   = jnp.zeros(seq_len, dtype=bool).at[indices[1:] - 1].set(True)
    
    # Общий шаг скана
    def scan_step(carry, inputs):
        val, is_reset = inputs
        # Если сброс -> новое значение = x, иначе рекуррентная формула
        new_val = jnp.where(is_reset, val, gamma * val + alpha * carry)
        return new_val, new_val
        
    # Прямой EMA (сброс на is_start)
    _, y = jax.lax.scan(scan_step, 0.0, (x, is_start))
    
    # Обратный EMA (переворачиваем массив и маску конца)
    x_rev = jnp.flip(x)
    is_end_rev = jnp.flip(is_end)  # Теперь True стоит на первом элементе перевёрнутого сегмента
    _, z_rev = jax.lax.scan(scan_step, 0.0, (x_rev, is_end_rev))
    z = jnp.flip(z_rev)
    
    return beta * y + (1.0 - beta) * z
    
bidir_ema_jax = jax.vmap(_process_row, in_axes=(0, None, None, None))

class MyAttentiveTopicModel:
    """
    Тематическая модель с двунаправленным attention на основе
    экспоненциального скользящего среднего.

    По смыслу:
    - ``phi[w, t]`` хранит распределение тем для слова ``w``;
    - ``p_ti`` хранит распределение тем для позиции ``i``;
    - ``theta_ti`` — контекстно-сглаженная версия ``p_ti`` после применения
      оператора attention внутри документа;
    - ``n_t`` и ``n_w`` играют роль глобальных эмпирических априоров по темам
      и словам.
    """

    def __init__(
            self,
            vocab_size: int,
            ctx_len: int,
            *,
            n_topics: int = 10,
            beta: float = 0.5,
            gamma_i: float = 0.6, #// коэффициент затухания будущих токенов
            gamma_n: float = 0.6, #// коэффициент затухания прошедших токенов
            attention_mode: str = 'ema',
            explicit_include_self: bool = True,
            weights: jax.Array | None = None,   #// двумерная матрица (I, 2C+1)
            n_attention_passes: int = 1,
            regularizers: list | None = None,
            alpha_regularizers: list[Callable[[jax.Array], jax.Array | float]] | None = None,
            metrics: list | None = None,
            eps: float = 1e-12,
    ):
        self.vocab_size = vocab_size
        self.ctx_len = ctx_len
        self.n_topics = n_topics
        self.beta = beta
        self.gamma_i = gamma_i
        self.gamma_n = gamma_n
        self.attention_mode = attention_mode
        self.explicit_include_self = explicit_include_self
        self.weights = None if weights is None else jnp.asarray(weights)
        self.n_attention_passes = n_attention_passes
        self._eps = eps
        self.phi = None
        self.n_t: jax.Array | None = None
        self.n_w: jax.Array | None = None
        self._validate_attention_config()

        self._regularizations = {}
        if regularizers is not None:
            for regularization in regularizers:
                self.add_regularization(regularization)

        self._alpha_regularizations = {}
        if alpha_regularizers is not None:
            for idx, regularization in enumerate(alpha_regularizers):
                self.add_alpha_regularization(tag=f'alpha_reg_{idx}', regularization=regularization)

        self._metrics = {}
        if metrics is not None:
            for metric in metrics:
                self.add_metric(metric)

    def _validate_attention_config(self):
        if self.attention_mode not in {'ema', 'explicit'}:
            raise ValueError(
                f'Unsupported attention_mode [{self.attention_mode}]. '
                f'Expected one of: ema, explicit.'
            )

        if self.attention_mode == 'ema':
            return

        if self.weights is None:
            return

        if self.weights.ndim != 2:
            raise ValueError('weights must be a 2D array with shape [n_tokens, 2K-1].')

        if self.weights.shape[1] == 0:
            raise ValueError('weights second dimension must be non-empty.')

        if self.weights.shape[1] % 2 == 0:
            raise ValueError(
                'weights length must be odd (2K-1): '
                '[past..., current, future...].'
            )

        self.weights = self.weights.astype(jnp.float32)

    def _explicit_directional_weights(
            self,
            *,
            weights_t: jax.Array,
            dtype,
    ) -> tuple[jax.Array, jax.Array]:
        # На входе веса для позиций текущего batch:
        # [past_far, ..., past_near, current, future_near, ..., future_far].
        # Для расчетов нужны коэффициенты вида [current, near, far].
        center = weights_t.shape[1] // 2
        center_weight = weights_t[:, center:center + 1] #// (I, 1)
        past_near_to_far = weights_t[:, :center][:, ::-1]
        future_near_to_far = weights_t[:, center + 1:]

        past = jnp.concatenate([center_weight, past_near_to_far], axis=1).astype(dtype) #// (I, C+1) веса от ведущего до самого левого
        future = jnp.concatenate([center_weight, future_near_to_far], axis=1).astype(dtype) #// (I, C+1) веса от ведущего до самого правого
        return past, future

    @partial(jax.jit, static_argnums=0)
    def _normalize_explicit_weights(self, weights_t: jax.Array) -> jax.Array:
        return self._norm_rows(weights_t) #// зачем нормализуются веса?

    @partial(jax.jit, static_argnums=0)
    def _effective_attention_weights(self, weights_t: jax.Array) -> jax.Array:
        # Для explicit_no_self центр исключается только в самом операторе
        # внимания: зануляем центральный коэффициент и нормируем соседей.
        if self.explicit_include_self:
            return self._norm_rows(weights_t)
        center = weights_t.shape[1] // 2
        masked = weights_t.at[:, center].set(0.0)
        return self._norm_rows(masked)

    @partial(jax.jit, static_argnums=(0,), static_argnames=('transpose',))
    def _apply_explicit_operator(
            self,
            *,
            x: jax.Array, #// x должен быть (2C+1, I)?
            weights_t: jax.Array,
            ctx_bounds: jax.Array,
            transpose: bool = False,
    ) -> jax.Array:
        if x.shape[1] == 0:
            return x

        if weights_t.shape[0] != x.shape[1]:
            raise ValueError(
                f'weights rows [{weights_t.shape[0]}] must match sequence length [{x.shape[1]}].'
            )

        #weights_t = self._effective_attention_weights(weights_t.astype(x.dtype)) #// зануляется ведущий и нормируются
        center = weights_t.shape[1] // 2
        #// две матрицы (I, C+1) веса от ведущего до самого левого и от ведущего до самого правого
        past_weights, future_weights = self._explicit_directional_weights(
            weights_t=weights_t,
            dtype=x.dtype,
        )
        #// ищем возможные позиции в ctx_bounds[1:] для (0, ..., 2С+1)
        #// получаем для каждой позиции номер документа, которой она принадлежит
        doc_ids = jnp.searchsorted(ctx_bounds[1:], jnp.arange(x.shape[1]), side='right')
        kernel_len = past_weights.shape[1] #// должно быть С+1

        if self.explicit_include_self:
            center_contrib = weights_t[:, center]
        else:
            center_contrib = jnp.zeros((x.shape[1],), dtype=x.dtype)
        result = center_contrib[None, :] * x #// непонятно - поэлементное умножение x либо на ноль, либо на веса ведущих 
        for offset in range(1, kernel_len): #// от 1 до С (полуокно)
            if x.shape[1] <= offset:
                break
            #// умножение x на веса окна со сдвигом от 1 до С с учетом границ документов
            same_doc = (doc_ids[offset:] == doc_ids[:-offset]).astype(x.dtype)
            if not transpose:
                result = result.at[:, offset:].add(
                    x[:, :-offset] * (past_weights[offset:, offset] * same_doc)[None, :]
                )
                result = result.at[:, :-offset].add(
                    x[:, offset:] * (future_weights[:-offset, offset] * same_doc)[None, :]
                )
            else:
                result = result.at[:, :-offset].add(
                    x[:, offset:] * (past_weights[offset:, offset] * same_doc)[None, :]
                )
                result = result.at[:, offset:].add(
                    x[:, :-offset] * (future_weights[:-offset, offset] * same_doc)[None, :]
                )

        return result

    def add_regularization(self, regularization: reg.Regularization):
        if not isinstance(regularization, reg.Regularization):
            regularization_name = getattr(regularization, '__name__', type(regularization).__name__)
            raise TypeError(
                f'Regularization [{regularization_name}] has to be a subclass of '
                f'the Regularization base class, got type {type(regularization)}'
            )

        self._regularizations[regularization.tag] = regularization

    def add_metric(self, metric: mtc.Metric):
        if not isinstance(metric, mtc.Metric):
            metric_name = getattr(metric, '__name__', type(metric).__name__)
            raise TypeError(
                f'Metric [{metric_name}] has to be a subclass of '
                f'the Metric base class, got type {type(metric)}'
            )

        self._metrics[metric.tag] = metric

    #//? что такое alpha_regularization?
    def add_alpha_regularization(self, *, tag: str, regularization: Callable[[jax.Array], jax.Array | float]):
        if not callable(regularization):
            raise TypeError(f'Alpha regularization [{tag}] must be callable.')
        self._alpha_regularizations[tag] = regularization

    def remove_regularization(self, tag: str):
        try:
            self._regularizations.pop(tag)
        except KeyError:
            print(
                f'Regularization with tag {tag} is not present. '
                f'Did you mean to use remove_metric?'
            )

    def remove_metric(self, tag: str):
        try:
            self._metrics.pop(tag)
        except KeyError:
            print(
                f'Metric with tag {tag} is not present. '
                f'Did you mean to use remove_regularization?'
            )

    def remove_alpha_regularization(self, tag: str):
        try:
            self._alpha_regularizations.pop(tag)
        except KeyError:
            print(f'Alpha regularization with tag {tag} is not present.')

    def _norm(self, x: jax.Array) -> jax.Array:
        return _norm_fn(x, self._eps)

    @partial(jax.jit, static_argnums=0)
    def _norm_rows(self, x: jax.Array) -> jax.Array:
        x = jnp.maximum(x, jnp.zeros_like(x))
        norm = x.sum(axis=1, keepdims=True)
        return jnp.where(norm > self._eps, x / norm, jnp.zeros_like(x))

    @partial(jax.jit, static_argnums=0)
    def _ema_windowed_attn(
            self,
            *,
            x: jax.Array,
            ctx_bounds: jax.Array,
    ) -> jax.Array:
        if x.shape[1] == 0:
            return x

        seq_len = x.shape[1] #// для p_ti.T это I
        doc_ids = jnp.searchsorted(ctx_bounds[1:], jnp.arange(seq_len), side='right') #// массив номеров документов для каждой позиции
        doc_starts = jnp.concatenate([ #// позиции начала документа
            jnp.array([True], dtype=bool),
            doc_ids[1:] != doc_ids[:-1],
        ])
        doc_ends = jnp.concatenate([ #// позиции за концом документа
            doc_ids[:-1] != doc_ids[1:],
            jnp.array([True], dtype=bool),
        ])

        forward = jnp.zeros_like(x)
        backward = jnp.zeros_like(x)

        # Ограничиваем влияние соседей окном длины ctx_len с каждой стороны.
        # При достаточном окне формула совпадает с EMA внутри документа.
        max_offset = self.ctx_len if self.ctx_len > 0 else seq_len - 1

        for offset in range(max_offset + 1):
            if seq_len <= offset:
                break

            decay_forward = (1.0 - self.gamma_i) ** offset
            decay_backward = (1.0 - self.gamma_n) ** offset

            if offset == 0:
                same_doc = jnp.ones((seq_len,), dtype=x.dtype)
                src_forward = x
                src_backward = x
                coeff_forward = decay_forward * (
                        self.gamma_i + (1.0 - self.gamma_i) * doc_starts.astype(x.dtype)
                )
                coeff_backward = decay_backward * (
                        self.gamma_n + (1.0 - self.gamma_n) * doc_ends.astype(x.dtype)
                )
                forward = forward + src_forward * coeff_forward[None, :]
                backward = backward + src_backward * coeff_backward[None, :]
                continue

            same_doc = (doc_ids[offset:] == doc_ids[:-offset]).astype(x.dtype)

            src_forward = x[:, :-offset]
            coeff_forward = decay_forward * (
                    self.gamma_i + (1.0 - self.gamma_i) * doc_starts[:-offset].astype(x.dtype)
            )
            forward = forward.at[:, offset:].add(
                src_forward * (coeff_forward * same_doc)[None, :]
            )

            src_backward = x[:, offset:]
            coeff_backward = decay_backward * (
                    self.gamma_n + (1.0 - self.gamma_n) * doc_ends[offset:].astype(x.dtype)
            )
            backward = backward.at[:, :-offset].add(
                src_backward * (coeff_backward * same_doc)[None, :]
            )

        return self.beta * forward + (1.0 - self.beta) * backward

    @partial(jax.jit, static_argnums=0)
    def _ema_windowed_attn_transpose(
            self,
            *,
            x: jax.Array,
            ctx_bounds: jax.Array,
    ) -> jax.Array:
        if x.shape[1] == 0:
            return x

        seq_len = x.shape[1]
        doc_ids = jnp.searchsorted(ctx_bounds[1:], jnp.arange(seq_len), side='right')
        doc_starts = jnp.concatenate([
            jnp.array([True], dtype=bool),
            doc_ids[1:] != doc_ids[:-1],
        ])
        doc_ends = jnp.concatenate([
            doc_ids[:-1] != doc_ids[1:],
            jnp.array([True], dtype=bool),
        ])

        forward_t = jnp.zeros_like(x)
        backward_t = jnp.zeros_like(x)
        max_offset = self.ctx_len if self.ctx_len > 0 else seq_len - 1

        for offset in range(max_offset + 1):
            if seq_len <= offset:
                break

            decay_forward = (1.0 - self.gamma_i) ** offset
            decay_backward = (1.0 - self.gamma_n) ** offset

            if offset == 0:
                same_doc = jnp.ones((seq_len,), dtype=x.dtype)
                coeff_forward = decay_forward * (
                        self.gamma_i + (1.0 - self.gamma_i) * doc_starts.astype(x.dtype)
                )
                coeff_backward = decay_backward * (
                        self.gamma_n + (1.0 - self.gamma_n) * doc_ends.astype(x.dtype)
                )
                forward_t = forward_t + x * (coeff_forward * same_doc)[None, :]
                backward_t = backward_t + x * (coeff_backward * same_doc)[None, :]
                continue

            same_doc = (doc_ids[offset:] == doc_ids[:-offset]).astype(x.dtype)

            coeff_forward = decay_forward * (
                    self.gamma_i + (1.0 - self.gamma_i) * doc_starts[:-offset].astype(x.dtype)
            )
            forward_t = forward_t.at[:, :-offset].add(
                x[:, offset:] * (coeff_forward * same_doc)[None, :]
            )

            coeff_backward = decay_backward * (
                    self.gamma_n + (1.0 - self.gamma_n) * doc_ends[offset:].astype(x.dtype)
            )
            backward_t = backward_t.at[:, offset:].add(
                x[:, :-offset] * (coeff_backward * same_doc)[None, :]
            )

        return self.beta * forward_t + (1.0 - self.beta) * backward_t

    @partial(jax.jit, static_argnums=0)
    def _attn_function(
            self,
            x: jax.Array,
            *,
            weights_t: jax.Array | None = None,
            ctx_bounds: jax.Array | None = None,
    ) -> jax.Array:
        # Базовый оператор attention для одной последовательности.
        # В режиме `ema` это двунаправленное EMA.
        # В режиме `explicit` веса задаются на уровне позиций:
        # `weights_t[pos, 2K-1]` для текущего документа/батча.
        if self.attention_mode == 'explicit':
            if weights_t is None:
                raise ValueError('Explicit attention requires token-level weights.')
            if ctx_bounds is None:
                ctx_bounds = jnp.array([0, x.shape[1]], dtype=int)
            return self._segmented_attn_function_explicit(
                x=x,
                weights_t=weights_t,
                ctx_bounds=ctx_bounds,
            )

        if ctx_bounds is None:
            ctx_bounds = jnp.array([0, x.shape[1]], dtype=int)
        return self._ema_windowed_attn(x=x, ctx_bounds=ctx_bounds)

    @partial(jax.jit, static_argnums=0)
    def _segmented_attn_function_explicit(
            self,
            *,
            x: jax.Array,
            weights_t: jax.Array,
            ctx_bounds: jax.Array,
    ) -> jax.Array:
        if weights_t is None:
            raise ValueError('Explicit attention requires token-level weights.')
        return self._apply_explicit_operator(
            x=x,
            weights_t=weights_t,
            ctx_bounds=ctx_bounds,
            transpose=False,
        )

    def _resolve_ctx_bounds(
            self,
            *,
            batch: jax.Array,
            ctx_bounds: jax.Array | None,
    ) -> jax.Array:
        # Если границы документов не переданы, считаем, что весь batch —
        # это один документ. Такой режим нужен и для простых экспериментов,
        # и для совместимости с обычным ARTM-сценарием без сегментации.
        if ctx_bounds is None:
            return jnp.array([0, len(batch)], dtype=int)
        return ctx_bounds

    def _resolve_explicit_weights(
            self,
            *,
            weights_t: jax.Array | None,
            batch: jax.Array,
    ) -> jax.Array | None:
        if self.attention_mode != 'explicit':
            return None

        if weights_t is None:
            if self.weights is None:
                raise ValueError(
                    'Explicit attention requires token-level weights with '
                    'shape [len(batch), 2K-1].'
                )
            weights_t = self.weights

        if weights_t.ndim != 2:
            raise ValueError('Explicit weights must be a 2D array [len(batch), 2K-1].')

        if weights_t.shape[0] != len(batch):
            raise ValueError(
                f'Explicit weights rows [{weights_t.shape[0]}] must match batch length [{len(batch)}].'
            )

        if weights_t.shape[1] % 2 == 0:
            raise ValueError('Explicit weights second dimension must be odd (2K-1).')

        return self._normalize_explicit_weights(weights_t.astype(jnp.float32))

    @partial(jax.jit, static_argnums=0)
    def _get_doc_ids(
            self,
            *,
            batch: jax.Array,
            ctx_bounds: jax.Array,
    ) -> jax.Array:
        # Для каждой позиции токена вычисляем индекс документа, которому она
        # принадлежит. Это позволяет затем выполнять attention по всей
        # склеенной последовательности, но с мягким "сбросом" состояния
        # на границах документов.
        return jnp.searchsorted(ctx_bounds[1:], jnp.arange(len(batch)), side='right')

    @partial(jax.jit, static_argnums=0)
    def _get_doc_start_end_flags(
            self,
            *,
            batch: jax.Array,
            ctx_bounds: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        # Булевы маски начала/конца документа используются вместо явного
        # цикла по документам. Это важная оптимизация: один `lax.scan`
        # по всему batch компилируется и векторизуется лучше, чем множество
        # коротких проходов по отдельным документам.
        doc_ids = self._get_doc_ids(batch=batch, ctx_bounds=ctx_bounds)
        doc_starts = jnp.concatenate([
            jnp.array([True], dtype=bool),
            doc_ids[1:] != doc_ids[:-1],
        ])
        doc_ends = jnp.concatenate([
            doc_ids[:-1] != doc_ids[1:],
            jnp.array([True], dtype=bool),
        ])
        return doc_starts, doc_ends

    @partial(jax.jit, static_argnums=0)
    def _segmented_attn_function(
            self,
            *,
            x: jax.Array,
            doc_starts: jax.Array,
            doc_ends: jax.Array,
            batch: jax.Array,
            ctx_bounds: jax.Array,
            weights_t: jax.Array | None = None,
    ) -> jax.Array:
        # Сегментированная версия attention: вычисление идет по одной длинной
        # последовательности, но состояние рекуррентного EMA обнуляется на
        # границах документов. Это эквивалентно циклу "для каждого документа",
        # только без Python-overhead и без потери преимуществ JIT-компиляции.
        if self.attention_mode == 'explicit':
            return self._segmented_attn_function_explicit(
                x=x,
                weights_t=weights_t,
                ctx_bounds=ctx_bounds,
            )

        return self._ema_windowed_attn(x=x, ctx_bounds=ctx_bounds)

    #// расчет theta_ti
    #// исходные данные
    @partial(jax.jit, static_argnums=0)
    def _calc_theta_ti(
            self,
            *,
            p_ti: jax.Array,    #// (I, T) текущие вероятности тем для каждой позиции
            batch: jax.Array,   #// (I,) массив токенов в батче 
            ctx_bounds: jax.Array,  #// (D + 1) границы документов (позиции старта и после финиша)
            weights_t: jax.Array | None = None,
    ) -> jax.Array:
        # Шаг 7 алгоритма: theta_ti = Attn(p_ti).
        # Это оптимизированный путь: внимание считается одним
        # segmented-scan по всему batch с обнулением состояния на
        # границах документов.
        #
        # Для бенчмарка с наивным циклом по документам см.
        # `_calc_theta_ti_naive`.
        if len(batch) == 0:
            raise ValueError('batch must not be empty')

        weights_t = self._resolve_explicit_weights(weights_t=weights_t, batch=batch)
        doc_starts, doc_ends = self._get_doc_start_end_flags(batch=batch, ctx_bounds=ctx_bounds)
        return self._segmented_attn_function(
            x=p_ti.T,
            doc_starts=doc_starts,  #//? не нужно передавать - считаются внутри
            doc_ends=doc_ends,      #//? не нужно передавать - считаются внутри
            batch=batch,
            ctx_bounds=ctx_bounds,
            weights_t=weights_t,
        ).T

    def _calc_theta_ti_naive(
            self,
            *,
            p_ti: jax.Array,
            batch: jax.Array,
            ctx_bounds: jax.Array,
            weights_t: jax.Array | None = None,
    ) -> jax.Array:
        # Наивная baseline-реализация для экспериментов по скорости.
        # В отличие от `_calc_theta_ti`, здесь есть явный Python-цикл
        # по документам: для каждого сегмента отдельно вызывается базовый
        # оператор `_attn_function`, а затем результат конкатенируется.
        if len(batch) == 0:
            raise ValueError('batch must not be empty')

        weights_t = self._resolve_explicit_weights(weights_t=weights_t, batch=batch)
        bounds_host = jax.device_get(ctx_bounds)
        theta_parts = []
        for start, end in zip(bounds_host[:-1], bounds_host[1:]):
            start_i = int(start)
            end_i = int(end)
            p_ti_segment = p_ti[start_i:end_i]
            segment_weights = None
            if self.attention_mode == 'explicit' and weights_t:
                segment_weights = weights_t[start_i:end_i]
            theta_parts.append(
                self._attn_function(
                    x=p_ti_segment.T,
                    weights_t=segment_weights,
                    ctx_bounds=jnp.array([0, end_i - start_i], dtype=int),
                ).T
            )

        return jnp.concatenate(theta_parts, axis=0)

    @partial(jax.jit, static_argnums=0)
    def _apply_attn_transpose(
            self,
            *,
            x: jax.Array,
            batch: jax.Array,
            ctx_bounds: jax.Array,
            weights_t: jax.Array | None = None,
    ) -> jax.Array:
        # Это один из самых важных оптимизационных узлов модели.
        #
        # В формуле для N_tw присутствует q_wi = Attn(1[w_i = w]), и в лоб
        # нужно было бы:
        # 1) построить attention для каждого уникального слова;
        # 2) умножить его на p_ti / theta_ti;
        # 3) суммировать вклад по позициям.
        #
        # Вместо этого используется линейность оператора Attn:
        # sum_i Attn(e_i) * x_i == Attn^T(x),
        # где e_i — индикаторные последовательности.
        #
        # Поэтому мы сразу применяем транспонированный оператор attention к
        # ratio = p_ti / theta_ti и потом одним scatter-add собираем N_tw.
        # Это снимает зависимость от числа уникальных слов в документе и
        # заметно сокращает объем промежуточных тензоров.
        if len(batch) == 0:
            raise ValueError('batch must not be empty')

        weights_t = self._resolve_explicit_weights(weights_t=weights_t, batch=batch)

        if self.attention_mode == 'explicit':
            # Для explicit-режима используем отдельный оптимизированный
            # оператор A^T, который учитывает позиционно-зависимые веса.
            return self._apply_explicit_operator(
                x=x,
                weights_t=weights_t,
                ctx_bounds=ctx_bounds,
                transpose=True,
            )

        return self._ema_windowed_attn_transpose(x=x, ctx_bounds=ctx_bounds)

    #// закончил здесь
    @partial(jax.jit, static_argnums=0)
    def _calc_q_local(
            self,
            *,
            batch: jax.Array,
            ctx_bounds: jax.Array,
            weights_t: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        # Явный расчет q_wi нужен в основном для тестов и диагностики.
        # В основном цикле обучения он не используется, потому что прямой
        # расчет через индикаторные последовательности дорог по памяти.
        if len(batch) == 0:
            raise ValueError('batch must not be empty')

        weights_t = self._resolve_explicit_weights(weights_t=weights_t, batch=batch)
        doc_starts, doc_ends = self._get_doc_start_end_flags(batch=batch, ctx_bounds=ctx_bounds)
        unique_tokens, inverse = jnp.unique(
            batch,
            return_inverse=True,
            size=len(batch),
            fill_value=-1,
        )
        unique_mask = unique_tokens >= 0
        indicators = jax.nn.one_hot(inverse, len(batch), dtype=jnp.float32).T
        q_local = self._segmented_attn_function(
            x=indicators,
            doc_starts=doc_starts,
            doc_ends=doc_ends,
            batch=batch,
            ctx_bounds=ctx_bounds,
            weights_t=weights_t,
        )
        q_local = jnp.where(unique_mask[:, None], q_local, 0.0)
        safe_tokens = jnp.where(unique_mask, unique_tokens, 0)
        return safe_tokens, q_local, unique_mask

    def _update_p_ti(
            self,
            *,
            values: jax.Array,
            theta_ti: jax.Array,
    ) -> jax.Array | None:
        # Шаг 8 алгоритма: уточняем p_ti пропорционально
        # values * theta_ti / n_t, затем нормируем по темам.
        # В зависимости от места вызова values может быть phi[w_i, :]
        # или уже предыдущее p_ti.
        if self.n_t is None:
            return None 
        p_t = self.n_t / jnp.maximum(jnp.sum(self.n_t), self._eps)
        p_ti = values * theta_ti / jnp.maximum(p_t[None, :], self._eps)
        return self._norm(p_ti.T).T

    def _calc_p_ti(
            self,
            *,
            phi: jax.Array,
            theta_ti: jax.Array,
            batch: jax.Array,
    ) -> jax.Array | None:
        phi_twi = phi[batch]
        return self._update_p_ti(values=phi_twi, theta_ti=theta_ti)

    @partial(jax.jit, static_argnums=0)
    def _calc_n_w(
            self,
            *,
            batch: jax.Array,
    ) -> jax.Array:
        # Глобальные частоты слов считаются один раз и затем используются
        # как в M-шаге, так и при вычислении perplexity.
        return jnp.bincount(
            batch,
            length=self.vocab_size,
            minlength=self.vocab_size,
        )

    # Подсчитать глобальные частоты слов
    def _init_word_counts(
            self,
            data: jax.Array,
    ):
        self.n_w = self._calc_n_w(batch=data).astype(jnp.float32)

    def _calc_p_wi(
            self,
            *,
            phi: jax.Array,
            theta_ti: jax.Array,
            batch: jax.Array,
    ) -> jax.Array:
        # Вероятность наблюдения слова в позиции i после интегрирования по
        # темам. Используется в perplexity и соответствует байесовской
        # декомпозиции p(w_i) * sum_t p(t | w_i) * p(t | i) / p(t).
        if self.n_w is None or self.n_t is None:
            raise ValueError('Model priors are not initialized.')

        p_t = self.n_t / jnp.maximum(jnp.sum(self.n_t), self._eps)
        p_w = self.n_w / jnp.maximum(jnp.sum(self.n_w), self._eps)
        return p_w[batch] * jnp.sum(
            phi[batch] * (theta_ti / jnp.maximum(p_t[None, :], self._eps)),
            axis=1,
        )

    @partial(jax.jit, static_argnums=0)
    def _calc_n_tw_simple(
            self,
            *,
            p_ti: jax.Array,
            batch: jax.Array,
    ) -> jax.Array:
        # Локальная статистика n_tw из шага 11: обычная сумма p_ti по всем
        # вхождениям слова w. Реализована через scatter-add.
        return jnp.add.at(
            jnp.zeros((self.vocab_size, self.n_topics), dtype=p_ti.dtype),
            batch,
            p_ti,
            inplace=False,
        )

    @partial(jax.jit, static_argnums=0)
    def _calc_n_tw(
            self,
            *,
            p_ti: jax.Array,
            theta_ti: jax.Array,
            batch: jax.Array,
            ctx_bounds: jax.Array,
            weights_t: jax.Array | None = None,
    ) -> jax.Array:
        # Контекстная статистика N_tw из шага 10.
        # Вместо явного q_wi сначала вычисляем ratio = p_ti / theta_ti,
        # затем применяем Attn^T и агрегируем вклад по словам.
        ratio = p_ti / jnp.maximum(theta_ti, self._eps)
        attn_ratio = self._apply_attn_transpose(
            x=ratio.T,
            batch=batch,
            ctx_bounds=ctx_bounds,
            weights_t=weights_t,
        ).T
        return jnp.add.at(
            jnp.zeros((self.vocab_size, self.n_topics), dtype=attn_ratio.dtype),
            batch,
            attn_ratio,
            inplace=False,
        )

    @partial(jax.jit, static_argnums=(0,), static_argnames=('grad_reg',))
    def _calc_phi(
            self,
            *,
            n_tw: jax.Array,
            N_tw: jax.Array,
            grad_reg: Callable,
            phi: jax.Array | None = None,
    ) -> jax.Array:
        # Шаг 12 алгоритма.
        # phi_base = n_tw / n_w соответствует текущей оценке phi без
        # нормировки, а затем к ней добавляются:
        # - контекстная поправка phi_base * N_tw;
        # - вклад регуляризаторов phi_base * grad_reg(phi_base).
        #
        # Умножение на phi_base удерживает обновление в той же шкале, что и
        # основная EM-статистика, и помогает избежать слишком агрессивных
        # скачков в редких словах.
        if self.n_w is None:
            raise ValueError('Word counts must be initialized before phi update.')

        phi_base = n_tw / jnp.maximum(self.n_w[:, None], self._eps)
        reg_term = grad_reg(phi_base)
        phi_new = n_tw + phi_base * N_tw + phi_base * reg_term
        #return self._norm(phi_new.T).T
        return self._norm(phi_new)

    @partial(jax.jit, static_argnums=(0,), static_argnames=('grad_alpha_reg',))
    def _calc_weights_ti(
            self,
            *,
            weights_t: jax.Array,
            p_ti: jax.Array,
            theta_ti: jax.Array,
            phi: jax.Array,
            batch: jax.Array,
            ctx_bounds: jax.Array,
            grad_alpha_reg: Callable,
    ) -> jax.Array:
        # Alpha-обновление для explicit-внимания:
        # alpha_ci <- norm_c(alpha_ci * [sum_t p_ti * phi_{t,w_c}/theta_ti + dS/dalpha_ci]).
        ratio = p_ti / jnp.maximum(theta_ti, self._eps)
        phi_batch = phi[batch]
        doc_ids = self._get_doc_ids(batch=batch, ctx_bounds=ctx_bounds)

        kernel_len = weights_t.shape[1]
        center = kernel_len // 2
        seq_len = len(batch)

        center_term = jnp.sum(ratio * phi_batch, axis=1, keepdims=True)
        past_near = jnp.zeros((seq_len, center), dtype=weights_t.dtype)
        future_near = jnp.zeros((seq_len, center), dtype=weights_t.dtype)

        for offset in range(1, center + 1):
            if seq_len <= offset:
                break

            same_doc = (doc_ids[offset:] == doc_ids[:-offset]).astype(weights_t.dtype)
            past_vals = jnp.sum(ratio[offset:] * phi_batch[:-offset], axis=1) * same_doc
            future_vals = jnp.sum(ratio[:-offset] * phi_batch[offset:], axis=1) * same_doc
            past_near = past_near.at[offset:, offset - 1].set(past_vals)
            future_near = future_near.at[:-offset, offset - 1].set(future_vals)

        factors = jnp.concatenate([past_near[:, ::-1], center_term, future_near], axis=1)
        reg_term = grad_alpha_reg(weights_t)
        updated = weights_t * (factors + reg_term)
        return self._normalize_explicit_weights(updated)

    def _compose_regularizations(self):
        # Регуляризаторы задаются как скалярные функции от phi, а здесь
        # собирается единый JAX-градиент. Благодаря этому регуляризация
        # встраивается в jit-компилируемый граф без ручного вывода формул.
        regs = self._regularizations.values()
        reg_grad = jax.grad(lambda x: sum([0.0, ] + [reg(x) for reg in regs]))
        return jax.jit(reg_grad)

    def _compose_alpha_regularizations(self):
        regs = self._alpha_regularizations.values()
        reg_grad = jax.grad(lambda x: sum([0.0, ] + [reg(x) for reg in regs]))
        return jax.jit(reg_grad)

    def _calc_metrics(
            self,
            *,
            phi_it: jax.Array,
            phi_wt: jax.Array,
            theta: jax.Array,
            verbose: int,
    ):
        if len(self._metrics) == 0:
            return

        if verbose > 1:
            print('  Metrics:')
        for tag, metric in self._metrics.items():
            value = metric(phi_it=phi_it, phi_wt=phi_wt, theta=theta)
            if verbose > 1:
                print(f'    {tag}: {value:.04f}')

    def _phase_count(self) -> int:
        base = 4 + self.n_attention_passes * 2
        if self.attention_mode == 'explicit':
            base += 1
        return base

    @staticmethod
    def _advance_phase_progress(progress, *, phase: str):
        if progress is None:
            return
        progress.set_postfix(phase=phase)
        progress.update(1)

    def _step(
            self,
            *,
            batch: jax.Array,
            phi: jax.Array,
            grad_reg: Callable,
            grad_alpha_reg: Callable,
            ctx_bounds: jax.Array | None = None,
            weights_t: jax.Array | None = None,
            phase_progress=None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array] | None:
        # Один полный EM-шаг по batch:
        # 1) инициализация p_ti из phi;
        # 2) несколько проходов attention/refinement;
        # 3) расчет статистик n_tw и N_tw;
        # 4) обновление phi и n_t.
        ctx_bounds = self._resolve_ctx_bounds(batch=batch, ctx_bounds=ctx_bounds)
        weights_t = self._resolve_explicit_weights(weights_t=weights_t, batch=batch)
        p_ti = phi[batch] #// шаг 5 (I, T) строки (распределение по темам) для токенов по порядку позиций в батче 
        #// стартовой вероятностью тем в позиции берем вероятности для токена в этой позиции из phi
        theta_ti = p_ti #// непонятная строка - theta_ti рассчитывается ниже в self._calc_theta_ti()

        for _ in range(self.n_attention_passes): # Шаг 6 алгоритма AARTM
            self._advance_phase_progress(phase_progress, phase='theta_refine')
            theta_ti = self._calc_theta_ti(
                p_ti=p_ti,
                batch=batch,
                ctx_bounds=ctx_bounds,
                weights_t=weights_t,
            )
            self._advance_phase_progress(phase_progress, phase='p_ti_refine')
            p_ti = self._update_p_ti(values=p_ti, theta_ti=theta_ti)

        self._advance_phase_progress(phase_progress, phase='n_tw')
        n_tw = self._calc_n_tw_simple(p_ti=p_ti, batch=batch)
        self._advance_phase_progress(phase_progress, phase='N_tw')
        N_tw = self._calc_n_tw(
            p_ti=p_ti,
            theta_ti=theta_ti,
            batch=batch,
            ctx_bounds=ctx_bounds,
            weights_t=weights_t,
        )
        self._advance_phase_progress(phase_progress, phase='phi_update')
        phi_new = self._calc_phi(
            n_tw=n_tw,
            N_tw=N_tw,
            grad_reg=grad_reg,
            phi=phi,
        )
        self._advance_phase_progress(phase_progress, phase='n_t')
        n_t_new = jnp.sum(p_ti, axis=0)
        weights_new = weights_t
        if self.attention_mode == 'explicit':
            self._advance_phase_progress(phase_progress, phase='alpha_update')
            weights_new = self._calc_weights_ti(
                weights_t=weights_t,
                p_ti=p_ti,
                theta_ti=theta_ti,
                phi=phi,
                batch=batch,
                ctx_bounds=ctx_bounds,
                grad_alpha_reg=grad_alpha_reg,
            )
        return p_ti, phi_new, theta_ti, n_t_new, weights_new

    def calc_perplexity(
            self,
            data: jax.Array,
            ctx_bounds: jax.Array,
            weights: jax.Array = None,
    ) -> float:
        # Для корректной perplexity повторяем тот же процесс inference для
        # p_ti/theta_ti, что и на обучении. Иначе метрика считалась бы для
        # другой модели, чем та, которая реально обучалась.
        if self.phi is None or self.n_t is None:
            raise ValueError('Model must be fitted before perplexity calculation.')

        if self.n_w is None:
            self._init_word_counts(data)

        log_likelihood = 0.0
        p_ti = self.phi[data]
        theta_ti = p_ti
        for _ in range(self.n_attention_passes):
            theta_ti = self._calc_theta_ti(
                p_ti=p_ti,
                batch=data,
                ctx_bounds=ctx_bounds,
                weights_t=weights,
            )
            p_ti = self._update_p_ti(values=p_ti, theta_ti=theta_ti)
        p_wi = self._calc_p_wi(phi=self.phi, theta_ti=theta_ti, batch=data)
        log_likelihood += float(jnp.sum(jnp.log(p_wi + self._eps)))
        return float(jnp.exp(-log_likelihood / len(data)))

    def fit(
            self,
            batch_data_with_doc_bounds: list[tuple[NDArray, NDArray]],
            *,
            max_iter: int = 1000,
            tol: float = 1e-3,
            seed: int = 42,
            gamma: float = 0.6
    ):
        # Инициализация близка к шагу 1 из алгоритма:
        # phi инициализируется случайно и нормируется по темам,
        # n_t стартует с равномерного вектора единиц.
        #key = jax.random.key(seed)
        np.random.seed(seed=seed)

        phi = _norm_numpy(np.random.uniform(
            size=(self.n_topics, self.vocab_size)),
            eps=self._eps
        )
        
        n_t = np.full(
            shape=(self.n_topics, ),
            fill_value=1.0,
        )
        grad_regularization = self._compose_regularizations()
        grad_alpha_regularization = self._compose_alpha_regularizations()

        for it in range(max_iter): #// шаг 2 - начало цикла проходов по всей коллекции
            #// шаг 3 инициализация
            n_w = np.zeros(self.vocab_size)
            n_tw: NDArray = np.zeros_like(phi)
            N_tw: NDArray = np.zeros_like(phi)
            n_t_tilda: NDArray = np.zeros_like(n_t)
            #// конец шаг 3 инициализация

            #// цикл для всех батчей шаги 4-11
            for batch, doc_bounds in batch_data_with_doc_bounds:

                #// шаг 5 
                p_ti: NDArray = phi[:, batch] #// шаг 5 (I, T) строки (распределение по темам) для токенов по порядку позиций в батче 

                #//здесь возможно вставить шаг 6 цикл для L блоков внимания

                #// шаг 7 
                theta_ti = bidir_ema(
                    X=p_ti, 
                    indices=doc_bounds, 
                    gamma=gamma, 
                    beta=0.5
                )

                #// шаг 8 
                p_ti = _norm_numpy(p_ti * theta_ti / n_t[:, None])

                #// шаг 9 - расчет q_wi
                wi_equal_w = (batch == np.arange(phi.shape[1])[:, None]).astype(float)
                q_wi = bidir_ema(
                    wi_equal_w, 
                    indices=doc_bounds, 
                    gamma=gamma, 
                    beta=0.5
                )

                #// шаг 10 
                N_tw += (p_ti / theta_ti) @ q_wi.T
                
                #// шаг 11.1 
                rows = np.arange(self.n_topics)[:, None]
                np.add.at(n_tw, (rows, batch), p_ti)

                #// шаг 11.2 
                n_t_tilda += np.sum(p_ti, axis=1)

                #// шаг ~11.3
                n_w += np.bincount(batch, minlength=phi.shape[1])

            #// конец цикл для всех батчей шаги 4-11

            phi_new = _norm_numpy(n_tw + np.divide(n_tw * N_tw, n_w, out=np.zeros_like(n_tw), where=n_w != 0))
            #phi_new = _norm_numpy(n_tw)
            n_t = n_t_tilda

            diff_norm = np.linalg.norm(phi_new - phi)
            jax.debug.print("Value of it: {it}", it=it)
            jax.debug.print("Value of diff_norm: {diff_norm}", diff_norm=diff_norm)
            '''self._calc_metrics(
                phi_it=phi_it,
                phi_wt=phi_new,
                theta=theta,
                verbose=verbose,
            )'''

            phi = phi_new
            if diff_norm < tol:
                self.phi = phi
                break

            #if it % 10 == 0:
            #    print(f"{it=}")

    def fit_my_jax(
            self,
            data: jax.Array,
            ctx_bounds: jax.Array,
            *,
            weights: jax.Array | None = None,
            max_iter: int = 1000,
            tol: float = 1e-3,
            verbose: int = 0,
            seed: int = 42,
            progress_bar: bool = False,
    ):

        key = jax.random.key(seed)

        gamma = self.gamma_i

        #// шаг 1 инициализация
        init_phi: jax.Array = jax.random.uniform(
                    key=key,
                    shape=(self.n_topics, self.vocab_size),
                    dtype=jnp.float64
                )
                
        self.phi = self._norm(init_phi)

        #self._init_word_counts(data)
        self.N_tw = jnp.zeros_like(self.phi)
        self.n_t = jnp.full(
            shape=(self.n_topics, ),
            fill_value=1.0,
        )
        #// конец шаг 1 инициализация
        grad_regularization = self._compose_regularizations()
        grad_alpha_regularization = self._compose_alpha_regularizations()

        phi = self.phi
        n_t = self.n_t
        n_w = jnp.zeros(self.vocab_size)

        iterator = range(max_iter)

        for it in iterator: #// шаг 2 - начало цикла проходов по всей коллекции
            #// шаг 3 инициализация
            n_tw: jax.Array = jnp.zeros_like(self.phi)
            N_tw: jax.Array = jnp.zeros_like(self.phi)
            n_t_tilda: jax.Array = jnp.zeros_like(self.n_t)
            #// конец шаг 3 инициализация

            #// цикл для всех батчей шаги 4-11
            for batch in data:

                #// шаг 5 
                p_ti: jax.Array = self.phi[batch] #// шаг 5 (I, T) строки (распределение по темам) для токенов по порядку позиций в батче 

                #//здесь возможно вставить шаг 6 цикл для L блоков внимания

                #// шаг 7 
                theta_ti = bidir_ema(
                    X=np.asarray(p_ti), 
                    indices=np.asarray(ctx_bounds), 
                    gamma=gamma, 
                    beta=0.5
                )

                #// шаг 8 
                p_ti = _norm_fn(p_ti * theta_ti / self.n_t[:, None])

                #// шаг 9 - расчет q_wi
                wi_equal_w = (batch == np.arange(phi)[:, None]).astype(float)
                q_wi = bidir_ema(
                    wi_equal_w, 
                    indices=np.asarray(ctx_bounds), 
                    gamma=gamma, 
                    beta=0.5
                )

                #// шаг 10 
                N_tw += q_wi * p_ti / theta_ti
                
                #// шаг 11.1 
                rows = np.arange(self.n_topics)[:, None]
                n_tw = jnp.add.at(n_tw, (rows, batch), p_ti)

                #// шаг 11.2 
                n_t_tilda += np.sum(p_ti, axis=1)

            #// конец цикл для всех батчей шаги 4-11

            phi_new = _norm_fn(n_tw + n_tw * N_tw / n_w)
            n_t = n_t_tilda

            diff_norm = jnp.linalg.norm(phi_new - phi)

            '''self._calc_metrics(
                phi_it=phi_it,
                phi_wt=phi_new,
                theta=theta,
                verbose=verbose,
            )'''

            phi = phi_new
            if diff_norm < tol:
                break

            if it % 10 == 00:
                print(f"{it=}")


    def fit_jax(
            self,
            batch_data_with_doc_bounds: list[tuple[Array, Array]],
            *,
            max_iter: int = 1000,
            tol: float = 1e-3,
            seed: int = 42,
            gamma: float = 0.6,
            beta: float = 0.5
    ):
        # Инициализация близка к шагу 1 из алгоритма:
        # phi инициализируется случайно и нормируется по темам,
        # n_t стартует с равномерного вектора единиц.
        key: Array = jax.random.key(seed)

        phi: Array = _norm_jax(jax.random.uniform(
            key=key,
            shape=(self.n_topics, self.vocab_size)),
            eps=self._eps
        )
        
        n_t: Array = jnp.full(
            shape=(self.n_topics, ),
            fill_value=1.0,
        )
        grad_regularization = self._compose_regularizations()
        grad_alpha_regularization = self._compose_alpha_regularizations()


        for it in range(max_iter): #// шаг 2 - начало цикла проходов по всей коллекции
            #// шаг 3 инициализация
            n_w: Array = jnp.zeros(self.vocab_size)
            n_tw: Array = jnp.zeros_like(phi)
            simple_N_tw: Array = jnp.zeros_like(phi)
            N_tw: Array = jnp.zeros_like(phi)
            n_t_tilda: Array = jnp.zeros_like(n_t)
            #// конец шаг 3 инициализация

            #// цикл для всех батчей шаги 4-11
            for batch, doc_bounds in batch_data_with_doc_bounds:

                #// шаг 5 
                p_ti: Array = phi[:, batch] #// шаг 5 (I, T) строки (распределение по темам) для токенов по порядку позиций в батче 

                #//здесь возможно вставить шаг 6 цикл для L блоков внимания

                #// шаг 7 
                theta_ti: Array = bidir_ema_jax(
                    p_ti, 
                    doc_bounds, 
                    gamma, 
                    beta
                )

                #// шаг 8 
                #p_ti = _norm_jax(p_ti * theta_ti / n_t[:, None])
                p_t = n_t / jnp.maximum(jnp.sum(n_t), self._eps)
                p_ti = _norm_jax(p_ti * theta_ti / jnp.maximum(p_t[:, None], self._eps))

                #// шаг 9 - расчет q_wi
                wi_equal_w: Array = (batch == jnp.arange(phi.shape[1])[:, None]).astype(float)
                q_wi = bidir_ema_jax(
                    wi_equal_w, 
                    doc_bounds, 
                    gamma, 
                    beta
                )

                #// шаг 10 
                #N_tw += (p_ti / theta_ti) @ q_wi.T
                N_tw += (p_ti / jnp.maximum(theta_ti, self._eps)) @ q_wi.T
                
                '''simple_N_tw += self._calc_n_tw(
                    p_ti=p_ti.T,
                    theta_ti=theta_ti.T,
                    batch=batch,
                    ctx_bounds=doc_bounds,
                    weights_t=None,
                ).T'''

                #print("Расхождение simple_N_tw и N_tw", jnp.allclose(simple_N_tw, N_tw).block_until_ready())
                #print(simple_N_tw[0])
                #print(N_tw[0])

                #// шаг 11.1 
                rows = jnp.arange(self.n_topics)[:, None]
                n_tw = jnp.add.at(n_tw, (rows, batch), p_ti, inplace=False)

                #// шаг 11.2 
                n_t_tilda += jnp.sum(p_ti, axis=1)

                #// шаг ~11.3
                if it < 1:
                    n_w += jnp.bincount(batch, minlength=phi.shape[1])

            #// конец цикл для всех батчей шаги 4-11

            if it < 1:
                self.n_w = n_w
                jax.debug.print("n_w = non_zero {sum_nz_n_w}", sum_nz_n_w=jnp.sum(self.n_w > 0))

            #phi_new = _norm_jax(n_tw)
            #phi_new = _norm_jax(n_tw + jnp.divide(n_tw * N_tw, n_w))
            #phi_new = _norm_jax(n_tw + n_tw * N_tw / jnp.maximum(n_w, self._eps))
            attn_lr = 1
            phi_new = _norm_jax(n_tw + attn_lr * jnp.divide(n_tw * N_tw, jnp.maximum(self.n_w, self._eps)))
            n_t = n_t_tilda

            diff_norm = jnp.linalg.norm(phi_new - phi)
            print(f"{it=}")
            print(f"{diff_norm=}")
            '''self._calc_metrics(
                phi_it=phi_it,
                phi_wt=phi_new,
                theta=theta,
                verbose=verbose,
            )'''

            phi = phi_new
            if diff_norm < tol:
                self.phi = phi
                break
