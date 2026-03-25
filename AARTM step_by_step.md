```python
from sys import path as syspath
from os import path as ospath

syspath.insert(1, r'D:\ARTM\topic-modelling-attention\src')
syspath
```




    ['C:\\Users\\kn\\AppData\\Roaming\\uv\\python\\cpython-3.13.9-windows-x86_64-none\\python313.zip',
     'D:\\ARTM\\topic-modelling-attention\\src',
     'C:\\Users\\kn\\AppData\\Roaming\\uv\\python\\cpython-3.13.9-windows-x86_64-none\\DLLs',
     'C:\\Users\\kn\\AppData\\Roaming\\uv\\python\\cpython-3.13.9-windows-x86_64-none\\Lib',
     'C:\\Users\\kn\\AppData\\Roaming\\uv\\python\\cpython-3.13.9-windows-x86_64-none',
     'C:\\Users\\kn\\AppData\\Local\\uv\\cache\\builds-v0\\.tmpGHKOtm',
     '',
     'C:\\Users\\kn\\AppData\\Local\\uv\\cache\\builds-v0\\.tmpGHKOtm\\Lib\\site-packages',
     'C:\\Users\\kn\\AppData\\Local\\uv\\cache\\archive-v0\\ZXpirby7L7XYy-rk4EZTc\\Lib\\site-packages',
     'C:\\Users\\kn\\AppData\\Local\\uv\\cache\\archive-v0\\ZXpirby7L7XYy-rk4EZTc\\Lib\\site-packages\\win32',
     'C:\\Users\\kn\\AppData\\Local\\uv\\cache\\archive-v0\\ZXpirby7L7XYy-rk4EZTc\\Lib\\site-packages\\win32\\lib',
     'C:\\Users\\kn\\AppData\\Local\\uv\\cache\\archive-v0\\ZXpirby7L7XYy-rk4EZTc\\Lib\\site-packages\\Pythonwin',
     'D:\\ARTM\\project_artm\\topic-modelling-attention\\.venv',
     'D:\\ARTM\\project_artm\\topic-modelling-attention\\.venv\\Lib\\site-packages',
     'D:\\ARTM\\project_artm\\topic-modelling-attention\\src',
     'D:\\ARTM\\project_artm\\topic-modelling-attention\\.venv\\Lib\\site-packages\\win32',
     'D:\\ARTM\\project_artm\\topic-modelling-attention\\.venv\\Lib\\site-packages\\win32\\lib',
     'D:\\ARTM\\project_artm\\topic-modelling-attention\\.venv\\Lib\\site-packages\\Pythonwin']




```python
import jax
import jax.numpy as jnp
import numpy as np 

from sklearn.datasets import fetch_20newsgroups

import matplotlib.pyplot as plt
import seaborn as sns

from cartm.model import ContextTopicModel
from cartm.preprocessing import DatasetPreprocessor
```

This example will guide you through the basic interaction with the model.

## Подготовка данных


```python
data = [
    'If I were you, I would try something new today.',
    'The best time for a new beginning is now.',
    'Small steps every day lead to big results.',
    'Sometimes you need to disconnect to reconnect.'
]
print(f'Total number of documents in corpus: {len(data)}')
print(f'Total number of words in corpus: {sum([len(doc.split(" ")) for doc in data])}')
```

    Total number of documents in corpus: 4
    Total number of words in corpus: 34
    


```python
preprocessor = DatasetPreprocessor()
tokenized_data, document_bounds = preprocessor.fit_transform(data)
print(f'Total number of document boundaries in preprocessed corpus: {len(document_bounds)}')
print(f'Total number of tokenized words in preprocessed corpus: {len(tokenized_data)}')
```

    Total number of document boundaries in preprocessed corpus: 5
    Total number of tokenized words in preprocessed corpus: 20
    


```python
vocabulary = preprocessor.vocabulary
print(document_bounds)
#sum(len(doc) for doc in tokenized_data)
print(tokenized_data)
print(vocabulary)
```

    [ 0  5  9 16 20]
    [18 17 12  8 16  1 15  8  0 11 14  5  3  6  2 10 13  7  4  9]
    {'begin': 0, 'best': 1, 'big': 2, 'day': 3, 'disconnect': 4, 'everi': 5, 'lead': 6, 'need': 7, 'new': 8, 'reconnect': 9, 'result': 10, 'small': 11, 'someth': 12, 'sometim': 13, 'step': 14, 'time': 15, 'today': 16, 'tri': 17, 'would': 18}
    

# Имитация итерации EM-алгоритма


```python
# функция фильтрации неотрицательных элементов и нормировки 
def _norm(x):
    _eps = 1e-12
    x = np.maximum(x, np.zeros_like(x))
    norm = x.sum(axis=0)
    x = np.where(norm > _eps, x / norm, np.zeros_like(x))
    return x
```

# Исходные параметры и переменные


```python
ctx_len = 3 # полуширина окна контекста
attn_bounds = document_bounds # границы документов - для обрезки окон контекста
print(f"{attn_bounds=}")
vocab_size = len(vocabulary) # размер словаря W
print(f"{vocab_size=}")
n_topics = 10 # количество топиков T 
```

    attn_bounds=Array([ 0,  5,  9, 16, 20], dtype=int32)
    vocab_size=19
    


```python
# матрица Фи размерность (W,T)
phi = jax.random.uniform(  # инициализация равномерным распределением 
            key=jax.random.key(42),
            shape=(vocab_size, n_topics),
        )
phi = _norm(phi) # нормировка
print(f"{phi.shape=}")
```

    phi.shape=(19, 10)
    


```python
# вектор-псевдоматрица размерностью (T, ) для хранения распределения топиков (частотности)
# сколько в корпусе (батче) токенов w приходится на топик t
n_t = jnp.full(           
            shape=(n_topics, ),
            fill_value=len(tokenized_data) / n_topics,    # инициализация средним значением
        )  # (T, )
print(f"{n_t=}")
```

    n_t=Array([2., 2., 2., 2., 2., 2., 2., 2., 2., 2.], dtype=float32, weak_type=True)
    

## Начало итерации (далее - шаг)

В результате шага необходимо получить пересчитанные значения для:
- phi_it - матрица вероятности контекста в позиции i при наличии в нем топика t 
- phi_new - матрица Фи 
- theta - матрица Тета 
- n_t - вектор частот топиков

Для случая пакетной обработки после каждого пакета корректируем phi_new и n_t_new с затуханием (lr = 0.1)

phi_new = phi_new * (1 - lr) + phi_step * lr # значение Фи, полученное после обработки пакета

n_t_new = n_t_new * (1 - lr) + n_t_step * lr # значение вектора частот топиков, полученное после обработки пакета



```python
# phi_hatch - матрица для перевзвешивания Фи с учетом частотности топиков (далее Фи^)
# phi.T умножается element-wise на n_t (частотность топиков), нормируется и транспонируется обратно
# (T, W) * (T, 1) = (T, W) -> t -> (W, T)
print(f"{phi.T.shape=}")
print(f"{n_t[:, None].shape=}")
phi_hatch = _norm(phi.T * n_t[:, None]).T # (W, T)
print(f"{phi_hatch.shape=}")

# в примере 19 строк для каждого токена с частотой соответствующего топика в 10 колонках
# токен 8 - "new", в векторе 10 значений частоты соответствующего топика
print(f"{phi_hatch[8]=}")
```

    phi.T.shape=(10, 19)
    n_t[:, None].shape=(10, 1)
    phi_hatch.shape=(19, 10)
    phi_hatch[8]=array([0.00623954, 0.00542419, 0.04397687, 0.14882576, 0.19154426,
           0.0731359 , 0.13844424, 0.06082839, 0.10824747, 0.22333333],
          dtype=float32)
    

### Далее расчет Теты с учетом перевзвешенной Фи^ и контекстов

Реализовано в функции 
```
_calc_theta(
    phi_hatch=phi_hatch,   # Фи^
    batch=batch,           # батч (массив токенов, обрабатываемых документов)
    ctx_bounds=ctx_bounds, # границы для обрезки окон контекстов (фактически границы документов)
    )
```


```python
batch = tokenized_data
print(f"{batch=}")
# выделенная часть Фи^, относящаяся к батчу - Фи^_батч
phi_it_hatch = jnp.take_along_axis( 
            phi_hatch,
            indices=batch[:, None],         # берем из Фи^ строки токенов, встречающихся в батче 
            axis=0,
            )
print(f"{phi_it_hatch.shape=}") # размерность матрицы Фи^_батч (позиции в батче Х топики) (C, T)
# phi_it_hatch - матрица 20 строк для каждой позиции (не токена) по порядку в батче (tokenized_data) с 10 колонками вероятности топиков
# пример: токен 8 - "new" стоит в 3 и 7 позиции в батче (tokenized_data)
print(f"{phi_it_hatch[3]=}")
print(f"{phi_it_hatch[7]=}")
```

    batch=Array([18, 17, 12,  8, 16,  1, 15,  8,  0, 11, 14,  5,  3,  6,  2, 10, 13,
            7,  4,  9], dtype=int32)
    phi_it_hatch.shape=(20, 10)
    phi_it_hatch[3]=Array([0.00623954, 0.00542419, 0.04397687, 0.14882576, 0.19154426,
           0.0731359 , 0.13844424, 0.06082839, 0.10824747, 0.22333333],      dtype=float32)
    phi_it_hatch[7]=Array([0.00623954, 0.00542419, 0.04397687, 0.14882576, 0.19154426,
           0.0731359 , 0.13844424, 0.06082839, 0.10824747, 0.22333333],      dtype=float32)
    

### Трехмерный тензор окон контекста 

Формирование для батча окон контекста для каждой позиции батча и топика


```python
def _get_context_tensor(batch):
    batch_size = batch.shape[0]     # количество позиций в батче B
    pad_token = -1  # assuming we don't have negative tokens in vocabulary
    
    # shifts for rolling the batch along new dimension
    #  [0, -1, -2, ..., -2 * ctx_len - 1]
    shifts = jnp.arange(0, -2 * ctx_len - 1, -1)  # (2C + 1, )
    print(f"{shifts=}")
    
    padding = jnp.full(
        (ctx_len, n_topics),
        fill_value=pad_token,
        dtype=batch.dtype,
    )  # (C, T)
    print(f"{padding.shape=}")
    print(f"{padding=}")
    
    padded_batch = jnp.concatenate([padding, batch, padding], axis=0)
    print(f"{padded_batch.shape=}")
    print(f"{padded_batch=}")
    
    # rolling and clipping each "slice" of batch
    def shift_batch(shift):
        return jnp.roll(padded_batch, shift, axis=0)[:batch_size]
    
    # apply vmap over all shifts
    # размножение батча с учетом контекстов - длина батча Х величина окна Х число топиков
    stacked_tensor = jax.vmap(shift_batch)(shifts).transpose(1, 0, 2)
    print(f"{stacked_tensor.shape=}")
    return stacked_tensor

# phi_it_hatch_with_context - Фи^_батч_контекст: трехмерный массив с измерениями 
#  {длина батча Х величина окна Х число топиков} (B x (2C+1) x T)
# где в ячейке вероятность, которую в заданной позиции i вносит токен из позиции j контекста Ci, для топика t
# окно контекста с учетом границ документов (зануляются токены вне границ)
# в примере получается (20 х 7 х 10)
phi_it_hatch_with_context = _get_context_tensor(batch=phi_it_hatch)
print(f"{phi_it_hatch_with_context.shape=}")
```

    shifts=Array([ 0, -1, -2, -3, -4, -5, -6], dtype=int32)
    padding.shape=(3, 10)
    padding=Array([[-1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
           [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
           [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1.]], dtype=float32)
    padded_batch.shape=(26, 10)
    padded_batch=Array([[-1.        , -1.        , -1.        , -1.        , -1.        ,
            -1.        , -1.        , -1.        , -1.        , -1.        ],
           [-1.        , -1.        , -1.        , -1.        , -1.        ,
            -1.        , -1.        , -1.        , -1.        , -1.        ],
           [-1.        , -1.        , -1.        , -1.        , -1.        ,
            -1.        , -1.        , -1.        , -1.        , -1.        ],
           [ 0.09118721,  0.11585489,  0.17614725,  0.15402782,  0.12468565,
             0.11698034,  0.04652762,  0.05900681,  0.08696249,  0.02861993],
           [ 0.1055353 ,  0.03661064,  0.05360207,  0.12943278,  0.15812047,
             0.14386164,  0.10507683,  0.11007395,  0.08883123,  0.06885507],
           [ 0.15339942,  0.09100751,  0.07821357,  0.02260839,  0.14305495,
             0.08793771,  0.13678788,  0.14964409,  0.08946344,  0.04788306],
           [ 0.00623954,  0.00542419,  0.04397687,  0.14882576,  0.19154426,
             0.0731359 ,  0.13844424,  0.06082839,  0.10824747,  0.22333333],
           [ 0.19601476,  0.06759968,  0.13736592,  0.12073773,  0.06520154,
             0.0299526 ,  0.13398725,  0.1092928 ,  0.08388831,  0.05595948],
           [ 0.09873395,  0.11448688,  0.14219065,  0.11818624,  0.08576075,
             0.03819812,  0.05098647,  0.13408248,  0.11187342,  0.10550101],
           [ 0.05242782,  0.13051964,  0.17815578,  0.00459592,  0.04665716,
             0.05834478,  0.09757855,  0.18038942,  0.08421864,  0.16711235],
           [ 0.00623954,  0.00542419,  0.04397687,  0.14882576,  0.19154426,
             0.0731359 ,  0.13844424,  0.06082839,  0.10824747,  0.22333333],
           [ 0.06033193,  0.16219935,  0.11991279,  0.14536451,  0.06596697,
             0.036351  ,  0.14481   ,  0.03493894,  0.06765784,  0.16246669],
           [ 0.15442893,  0.12945023,  0.03816544,  0.1207628 ,  0.0817277 ,
             0.1064999 ,  0.09887437,  0.10807201,  0.06045933,  0.10155922],
           [ 0.04113523,  0.16347514,  0.05199071,  0.05225467,  0.19773541,
             0.05192148,  0.02501237,  0.0742321 ,  0.18881956,  0.15342331],
           [ 0.02599362,  0.1455059 ,  0.06748758,  0.01224917,  0.14421405,
             0.15773168,  0.13484271,  0.08955493,  0.13365275,  0.08876758],
           [ 0.08824779,  0.03018737,  0.15288462,  0.18546036,  0.02518201,
             0.02654834,  0.01665173,  0.17318344,  0.198761  ,  0.10289332],
           [ 0.0217951 ,  0.03425077,  0.08293212,  0.16173272,  0.12125629,
             0.17084368,  0.08406883,  0.13451195,  0.07195906,  0.11664945],
           [ 0.00609265,  0.1556694 ,  0.15900597,  0.01569199,  0.11763589,
             0.17958316,  0.13261956,  0.08583287,  0.00244695,  0.14542158],
           [ 0.18575457,  0.13613965,  0.09336638,  0.09920645,  0.00283803,
             0.02944264,  0.17360304,  0.11124492,  0.11161297,  0.05679136],
           [ 0.16248712,  0.05080883,  0.18629105,  0.10292595,  0.13070355,
             0.07733082,  0.13189884,  0.07540288,  0.03239772,  0.04975311],
           [ 0.15727392,  0.11278932,  0.00512741,  0.14719586,  0.07000106,
             0.15478006,  0.14142025,  0.01202705,  0.16807547,  0.03130949],
           [ 0.17379853,  0.02632627,  0.16195871,  0.08363117,  0.09653858,
             0.2125369 ,  0.07256036,  0.00324063,  0.15490274,  0.01450603],
           [ 0.11185215,  0.17087771,  0.00998474,  0.07562889,  0.00351288,
             0.15623322,  0.01241871,  0.16161408,  0.09559387,  0.20228371],
           [-1.        , -1.        , -1.        , -1.        , -1.        ,
            -1.        , -1.        , -1.        , -1.        , -1.        ],
           [-1.        , -1.        , -1.        , -1.        , -1.        ,
            -1.        , -1.        , -1.        , -1.        , -1.        ],
           [-1.        , -1.        , -1.        , -1.        , -1.        ,
            -1.        , -1.        , -1.        , -1.        , -1.        ]],      dtype=float32)
    stacked_tensor.shape=(20, 7, 10)
    phi_it_hatch_with_context.shape=(20, 7, 10)
    

### Формирование матрицы внимания


```python
batch_size = len(tokenized_data)
# матрица размерностью (len(data) + 2C, 2C + 1)
attn_matrix = jnp.ones(
    shape=(batch_size + ctx_len * 2, ctx_len * 2 + 1),
    dtype=bool,
)  # 
# в примере (20 + 2 * 3, 2 * 3 )
print(attn_matrix.shape) 
```

    (26, 7)
    


```python
# формирование префиксных масок
prefix_bounds = document_bounds[: -1] + ctx_len
print(prefix_bounds)

ignored_mask_prefix = jnp.ones((ctx_len, ctx_len), dtype=bool)  # (C, C)
ignored_mask_prefix = jnp.rot90(~jnp.triu(ignored_mask_prefix))
print(ignored_mask_prefix)
ignored_mask_prefix = jnp.tile(ignored_mask_prefix, reps=len(prefix_bounds),).T
ignored_mask_prefix
```

    [ 3  8 12 19]
    [[False False False]
     [False False  True]
     [False  True  True]]
    




    Array([[False, False, False],
           [False, False,  True],
           [False,  True,  True],
           [False, False, False],
           [False, False,  True],
           [False,  True,  True],
           [False, False, False],
           [False, False,  True],
           [False,  True,  True],
           [False, False, False],
           [False, False,  True],
           [False,  True,  True]], dtype=bool)




```python
shifts = jnp.ones((len(prefix_bounds), ctx_len), dtype=int)  # (B, C)
print(shifts)
shifts = shifts.at[:, 0].set(prefix_bounds)
print(shifts)
shifts = jnp.cumsum(shifts, axis=1)
print(shifts)
shifts = shifts.reshape(-1, 1)  # (B * C, 1)
print(shifts.T)
```

    [[1 1 1]
     [1 1 1]
     [1 1 1]
     [1 1 1]]
    [[ 3  1  1]
     [ 8  1  1]
     [12  1  1]
     [19  1  1]]
    [[ 3  4  5]
     [ 8  9 10]
     [12 13 14]
     [19 20 21]]
    [[ 3  4  5  8  9 10 12 13 14 19 20 21]]
    


```python
prefix_columns = jnp.arange(ctx_len)  # (C, )
print(prefix_columns)
attn_matrix = attn_matrix.at[shifts, prefix_columns].set(ignored_mask_prefix)
print(attn_matrix.shape)
```

    [0 1 2]
    (26, 7)
    


```python
# формирование суффиксных масок
suffix_bounds = attn_bounds[1:]  # (B, )
ignored_mask_suffix = jnp.ones((ctx_len, ctx_len), dtype=bool)  # (C, C)
print(ignored_mask_suffix)
ignored_mask_suffix = jnp.rot90(~jnp.tril(ignored_mask_suffix))  # (C, C)
# for broadcasting
print(ignored_mask_suffix)
ignored_mask_suffix = jnp.tile(
    ignored_mask_suffix,
    reps=len(suffix_bounds),
).T  # (B * C, C)
print(ignored_mask_suffix)
```

    [[ True  True  True]
     [ True  True  True]
     [ True  True  True]]
    [[ True  True False]
     [ True False False]
     [False False False]]
    [[ True  True False]
     [ True False False]
     [False False False]
     [ True  True False]
     [ True False False]
     [False False False]
     [ True  True False]
     [ True False False]
     [False False False]
     [ True  True False]
     [ True False False]
     [False False False]]
    


```python
shifts = jnp.ones((len(suffix_bounds), ctx_len), dtype=int)  # (I, C)
print(shifts)
shifts = shifts.at[:, 0].set(suffix_bounds)
print(shifts)
shifts = jnp.cumsum(shifts, axis=1)
print(shifts)
shifts = shifts.reshape(-1, 1)  # (B * C, 1)
print(shifts.T)
```

    [[1 1 1]
     [1 1 1]
     [1 1 1]
     [1 1 1]]
    [[ 5  1  1]
     [ 9  1  1]
     [16  1  1]
     [20  1  1]]
    [[ 5  6  7]
     [ 9 10 11]
     [16 17 18]
     [20 21 22]]
    [[ 5  6  7  9 10 11 16 17 18 20 21 22]]
    


```python
suffix_columns = jnp.arange(ctx_len + 1, ctx_len * 2 + 1)  # (C, )
suffix_columns
```




    Array([4, 5, 6], dtype=int32)




```python
ignored_mask_suffix[::-1]
```




    Array([[False, False, False],
           [ True, False, False],
           [ True,  True, False],
           [False, False, False],
           [ True, False, False],
           [ True,  True, False],
           [False, False, False],
           [ True, False, False],
           [ True,  True, False],
           [False, False, False],
           [ True, False, False],
           [ True,  True, False]], dtype=bool)




```python
attn_matrix = attn_matrix.at[shifts[::-1], suffix_columns].set(ignored_mask_suffix[::-1])
attn_matrix
```




    Array([[ True,  True,  True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True,  True,  True],
           [False, False, False,  True,  True,  True,  True],
           [False, False,  True,  True,  True,  True,  True],
           [False,  True,  True,  True,  True,  True, False],
           [ True,  True,  True,  True,  True, False, False],
           [ True,  True,  True,  True, False, False, False],
           [False, False, False,  True,  True,  True,  True],
           [False, False,  True,  True,  True,  True, False],
           [False,  True,  True,  True,  True, False, False],
           [ True,  True,  True,  True, False, False, False],
           [False, False, False,  True,  True,  True,  True],
           [False, False,  True,  True,  True,  True,  True],
           [False,  True,  True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True,  True, False],
           [ True,  True,  True,  True,  True, False, False],
           [ True,  True,  True,  True, False, False, False],
           [False, False, False,  True,  True,  True,  True],
           [False, False,  True,  True,  True,  True, False],
           [False,  True,  True,  True,  True, False, False],
           [ True,  True,  True,  True, False, False, False],
           [ True,  True,  True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True,  True,  True]], dtype=bool)




```python
# remove padding
attn_matrix = attn_matrix[ctx_len: -ctx_len]  # (I, 2C + 1)
attn_matrix
```




    Array([[False, False, False,  True,  True,  True,  True],
           [False, False,  True,  True,  True,  True,  True],
           [False,  True,  True,  True,  True,  True, False],
           [ True,  True,  True,  True,  True, False, False],
           [ True,  True,  True,  True, False, False, False],
           [False, False, False,  True,  True,  True,  True],
           [False, False,  True,  True,  True,  True, False],
           [False,  True,  True,  True,  True, False, False],
           [ True,  True,  True,  True, False, False, False],
           [False, False, False,  True,  True,  True,  True],
           [False, False,  True,  True,  True,  True,  True],
           [False,  True,  True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True,  True, False],
           [ True,  True,  True,  True,  True, False, False],
           [ True,  True,  True,  True, False, False, False],
           [False, False, False,  True,  True,  True,  True],
           [False, False,  True,  True,  True,  True, False],
           [False,  True,  True,  True,  True, False, False],
           [ True,  True,  True,  True, False, False, False]], dtype=bool)




```python
print(document_bounds)
list(enumerate(list(attn_matrix)))
```

    [ 0  5  9 16 20]
    




    [(0, Array([False, False, False,  True,  True,  True,  True], dtype=bool)),
     (1, Array([False, False,  True,  True,  True,  True,  True], dtype=bool)),
     (2, Array([False,  True,  True,  True,  True,  True, False], dtype=bool)),
     (3, Array([ True,  True,  True,  True,  True, False, False], dtype=bool)),
     (4, Array([ True,  True,  True,  True, False, False, False], dtype=bool)),
     (5, Array([False, False, False,  True,  True,  True,  True], dtype=bool)),
     (6, Array([False, False,  True,  True,  True,  True, False], dtype=bool)),
     (7, Array([False,  True,  True,  True,  True, False, False], dtype=bool)),
     (8, Array([ True,  True,  True,  True, False, False, False], dtype=bool)),
     (9, Array([False, False, False,  True,  True,  True,  True], dtype=bool)),
     (10, Array([False, False,  True,  True,  True,  True,  True], dtype=bool)),
     (11, Array([False,  True,  True,  True,  True,  True,  True], dtype=bool)),
     (12, Array([ True,  True,  True,  True,  True,  True,  True], dtype=bool)),
     (13, Array([ True,  True,  True,  True,  True,  True, False], dtype=bool)),
     (14, Array([ True,  True,  True,  True,  True, False, False], dtype=bool)),
     (15, Array([ True,  True,  True,  True, False, False, False], dtype=bool)),
     (16, Array([False, False, False,  True,  True,  True,  True], dtype=bool)),
     (17, Array([False, False,  True,  True,  True,  True, False], dtype=bool)),
     (18, Array([False,  True,  True,  True,  True, False, False], dtype=bool)),
     (19, Array([ True,  True,  True,  True, False, False, False], dtype=bool))]




```python
def _get_context_weights_1d(gamma: float) -> np.ndarray:
    # значение из класса по умолчанию
    _self_aware_context = False
    # w_i = gamma * (1 - gamma)**i
    # правая половина весов контекста размером C
    suffix_context_weights = np.cumprod(np.full(ctx_len, (1 - gamma))) * gamma  # (C, ) 
    # заполняем ndarray длины self.ctx_len значениями (1 - gamma)
    # кумулятивное произведение [1, 2, 3, 4, 5] -> [1, 2, 6, 24, 120]
    #jax.debug.print("{suffix_context_weights}", suffix_context_weights=suffix_context_weights)
    prefix_context_weights = suffix_context_weights[::-1]  # (C, ) левая половина - перевернутая правая
    self_context_weight = np.array([gamma * _self_aware_context]) # массив из одного элемента _gamma или 0
    context_weights = np.concatenate([
        prefix_context_weights,
        self_context_weight,
        suffix_context_weights,
    ])
    # собранный массив типа ctx_len = 3, gamma = 0.6 -> [0.0384, 0.096 , 0.24  , 0.    , 0.24  , 0.096 , 0.0384]
    return context_weights  # (2C + 1, ) 

# при ctx_len = 3, gamma = 0.6
_context_weights_1d = _get_context_weights_1d(0.6)
print(f"{_context_weights_1d=}")
context_matrix = _context_weights_1d * attn_matrix  # (I, 2C + 1)
list(enumerate(list(context_matrix)))
```

    _context_weights_1d=array([0.0384, 0.096 , 0.24  , 0.    , 0.24  , 0.096 , 0.0384])
    




    [(0,
      Array([0.    , 0.    , 0.    , 0.    , 0.24  , 0.096 , 0.0384], dtype=float32)),
     (1,
      Array([0.    , 0.    , 0.24  , 0.    , 0.24  , 0.096 , 0.0384], dtype=float32)),
     (2, Array([0.   , 0.096, 0.24 , 0.   , 0.24 , 0.096, 0.   ], dtype=float32)),
     (3,
      Array([0.0384, 0.096 , 0.24  , 0.    , 0.24  , 0.    , 0.    ], dtype=float32)),
     (4,
      Array([0.0384, 0.096 , 0.24  , 0.    , 0.    , 0.    , 0.    ], dtype=float32)),
     (5,
      Array([0.    , 0.    , 0.    , 0.    , 0.24  , 0.096 , 0.0384], dtype=float32)),
     (6, Array([0.   , 0.   , 0.24 , 0.   , 0.24 , 0.096, 0.   ], dtype=float32)),
     (7, Array([0.   , 0.096, 0.24 , 0.   , 0.24 , 0.   , 0.   ], dtype=float32)),
     (8,
      Array([0.0384, 0.096 , 0.24  , 0.    , 0.    , 0.    , 0.    ], dtype=float32)),
     (9,
      Array([0.    , 0.    , 0.    , 0.    , 0.24  , 0.096 , 0.0384], dtype=float32)),
     (10,
      Array([0.    , 0.    , 0.24  , 0.    , 0.24  , 0.096 , 0.0384], dtype=float32)),
     (11,
      Array([0.    , 0.096 , 0.24  , 0.    , 0.24  , 0.096 , 0.0384], dtype=float32)),
     (12,
      Array([0.0384, 0.096 , 0.24  , 0.    , 0.24  , 0.096 , 0.0384], dtype=float32)),
     (13,
      Array([0.0384, 0.096 , 0.24  , 0.    , 0.24  , 0.096 , 0.    ], dtype=float32)),
     (14,
      Array([0.0384, 0.096 , 0.24  , 0.    , 0.24  , 0.    , 0.    ], dtype=float32)),
     (15,
      Array([0.0384, 0.096 , 0.24  , 0.    , 0.    , 0.    , 0.    ], dtype=float32)),
     (16,
      Array([0.    , 0.    , 0.    , 0.    , 0.24  , 0.096 , 0.0384], dtype=float32)),
     (17, Array([0.   , 0.   , 0.24 , 0.   , 0.24 , 0.096, 0.   ], dtype=float32)),
     (18, Array([0.   , 0.096, 0.24 , 0.   , 0.24 , 0.   , 0.   ], dtype=float32)),
     (19,
      Array([0.0384, 0.096 , 0.24  , 0.    , 0.    , 0.    , 0.    ], dtype=float32))]




```python
# нормирование матрицы весов контекста
# количество строк - позиции в батче 
# в каждой строке нормированные веса с учетом границ документов
context_matrix = _norm(context_matrix.T).T
print(f"{context_matrix=}") 
```

    context_matrix=array([[0.        , 0.        , 0.        , 0.        , 0.64102566,
            0.25641027, 0.10256411],
           [0.        , 0.        , 0.390625  , 0.        , 0.390625  ,
            0.15625001, 0.06250001],
           [0.        , 0.14285715, 0.35714287, 0.        , 0.35714287,
            0.14285715, 0.        ],
           [0.0625    , 0.15625   , 0.39062497, 0.        , 0.39062497,
            0.        , 0.        ],
           [0.1025641 , 0.25641024, 0.6410256 , 0.        , 0.        ,
            0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.64102566,
            0.25641027, 0.10256411],
           [0.        , 0.        , 0.4166667 , 0.        , 0.4166667 ,
            0.16666667, 0.        ],
           [0.        , 0.16666667, 0.4166667 , 0.        , 0.4166667 ,
            0.        , 0.        ],
           [0.1025641 , 0.25641024, 0.6410256 , 0.        , 0.        ,
            0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.64102566,
            0.25641027, 0.10256411],
           [0.        , 0.        , 0.390625  , 0.        , 0.390625  ,
            0.15625001, 0.06250001],
           [0.        , 0.13513514, 0.33783785, 0.        , 0.33783785,
            0.13513514, 0.05405406],
           [0.05128205, 0.12820512, 0.3205128 , 0.        , 0.3205128 ,
            0.12820512, 0.05128205],
           [0.05405405, 0.13513513, 0.33783782, 0.        , 0.33783782,
            0.13513513, 0.        ],
           [0.0625    , 0.15625   , 0.39062497, 0.        , 0.39062497,
            0.        , 0.        ],
           [0.1025641 , 0.25641024, 0.6410256 , 0.        , 0.        ,
            0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.64102566,
            0.25641027, 0.10256411],
           [0.        , 0.        , 0.4166667 , 0.        , 0.4166667 ,
            0.16666667, 0.        ],
           [0.        , 0.16666667, 0.4166667 , 0.        , 0.4166667 ,
            0.        , 0.        ],
           [0.1025641 , 0.25641024, 0.6410256 , 0.        , 0.        ,
            0.        , 0.        ]], dtype=float32)
    


```python
print(f"{context_matrix[..., None].shape=}")
print(f"{phi_it_hatch_with_context.shape=}")
```

    context_matrix[..., None].shape=(20, 7, 1)
    phi_it_hatch_with_context.shape=(20, 7, 10)
    


```python
# element-wise умножение матрицы весов контекста для каждой позиции (20, 7) на
# phi_it_hatch_with_context - трехмерный массив с измерениями длина батча Х величина окна Х число топиков (I Х 2C+1 Х T) (20, 7, 10)
# где в ячейке вероятность, которую в заданной позиции i вносит токен из позиции j в контексте Ci, для топика t
# окно контекста с учетом границ документов (зануляются токены вне границ)
# после умножения вероятности, взвешенные по весам в контексте
theta_it = context_matrix[..., None] * phi_it_hatch_with_context
print(f"{theta_it.shape=}")
```

    theta_it.shape=(20, 7, 10)
    


```python
# вот как умнножаются, чтобы получилась theta_it
print(f"{context_matrix[..., None][0][4]=}")
print(f"{phi_it_hatch_with_context[0][4]=}")
print(f"{theta_it[0][4]=}")
print(f"{context_matrix[..., None][0][4] * phi_it_hatch_with_context[0][4]=}")
```

    context_matrix[..., None][0][4]=array([0.64102566], dtype=float32)
    phi_it_hatch_with_context[0][4]=Array([0.1055353 , 0.03661064, 0.05360207, 0.12943278, 0.15812047,
           0.14386164, 0.10507683, 0.11007395, 0.08883123, 0.06885507],      dtype=float32)
    theta_it[0][4]=Array([0.06765083, 0.02346836, 0.0343603 , 0.08296973, 0.10135928,
           0.092219  , 0.06735694, 0.07056022, 0.0569431 , 0.04413787],      dtype=float32)
    context_matrix[..., None][0][4] * phi_it_hatch_with_context[0][4]=Array([0.06765083, 0.02346836, 0.0343603 , 0.08296973, 0.10135928,
           0.092219  , 0.06735694, 0.07056022, 0.0569431 , 0.04413787],      dtype=float32)
    


```python
# суммирование по измерению окна контекста - остаются позиции в батче и топики
# в результате вероятность топика в данной позиции, полученная по токенам в окне контекста 
theta_it = jnp.sum(theta_it, axis=1)  # (I, T)
print(f"{theta_it.shape=}")
```

    theta_it.shape=(20, 10)
    

## Расчет матрицы распределения вероятности топика для контекстов в позициях p_ti


```python
def _calc_p_ti(
        *,
        phi: jax.Array,
        theta: jax.Array,
        batch: jax.Array
) -> tuple[jax.Array, jax.Array]:
    # выделенная часть Фи, относящаяся к батчу - Фи_батч
    phi_it = jnp.take_along_axis(
        phi,
        indices=batch[:, None],         # берем из Фи строки токенов, встречающихся в батче 
        axis=0,
    )  # (I, T)
    print(f"{phi_it.shape=}")
    # element-wise умножение Фи (в части токенов, относящихся к батчу) на Тету
    # обе матрицы размерностью ((I, T))
    # нормализация по строкам - вероятность топиков для каждой позиции в сумме единица
    print(f"{theta.shape=}")
    print(f"{(phi_it * theta).shape=}")
    p_ti = _norm((phi_it * theta).T).T  # (I, T)
    print(f"{p_ti.shape=}")
    print()
    print("# element-wise")
    print(f"{phi_it[0]=}")
    print(f"{theta[0]=}")
    print(f"{(phi_it * theta)[0]=}")
    
    print(f"{phi_it[0][1]=}")
    print(f"{theta[0][1]=}")
    print(f"{(phi_it * theta)[0][1]=}")
    print(f"{phi_it[0][1] * theta[0][1]=}")
    
    return p_ti, phi_it

# phi_it - Фи (в части токенов, относящихся к батчу) вероятность темы при наличии токена w в позиции i
# theta_it - Тета вероятность топика в данной позиции, полученная по токенам в окне контекста позиции i
# p_ti - вероятность топика в этой позиции при наличии этого токена и окружающего его контекста
p_ti, phi_it = _calc_p_ti(
    phi=phi,
    theta=theta_it,
    batch=batch,
)  # (I, T)

```

    phi_it.shape=(20, 10)
    theta.shape=(20, 10)
    (phi_it * theta).shape=(20, 10)
    p_ti.shape=(20, 10)
    
    # element-wise
    phi_it[0]=Array([0.04269939, 0.05425029, 0.08248284, 0.07212517, 0.05838539,
           0.05477729, 0.02178705, 0.02763057, 0.04072112, 0.01340159],      dtype=float32)
    theta[0]=Array([0.10762397, 0.04735994, 0.05892551, 0.10403094, 0.15768561,
           0.12226825, 0.11663017, 0.11516932, 0.09098475, 0.07932156],      dtype=float32)
    (phi_it * theta)[0]=Array([0.00459548, 0.00256929, 0.00486034, 0.00750325, 0.00920654,
           0.00669752, 0.00254103, 0.00318219, 0.003705  , 0.00106303],      dtype=float32)
    phi_it[0][1]=Array(0.05425029, dtype=float32)
    theta[0][1]=Array(0.04735994, dtype=float32)
    (phi_it * theta)[0][1]=Array(0.00256929, dtype=float32)
    phi_it[0][1] * theta[0][1]=Array(0.00256929, dtype=float32)
    

### Оценка частоты топиков n_t на шаге


```python
# суммируем вероятности топика по всем позициям, получаем оценку частот топиков в батче
def _calc_n_t(*, p_ti):
    return jnp.sum(p_ti, axis=0)  # (T, )

n_t_new = _calc_n_t(p_ti=p_ti)
print(f"{n_t_new=}")
```

    n_t_new=Array([1.7109293, 1.5929288, 1.7314843, 1.9040339, 2.2407615, 2.2293854,
           2.1159027, 1.8696146, 2.1715155, 2.4334447], dtype=float32)
    

#### Подготовка регуляризатора для пересчета Фи


```python
# обработка регуляризаторов на примере DecorrelationRegularization
def decorrelation_reg(phi_wt: jax.Array) -> float:
    corr_matrix = phi_wt.T @ phi_wt  # (T, T)
    # remove duplicates and diagonal terms
    corr_triu = jnp.triu(corr_matrix, k=1)
    return jnp.sum(corr_triu)

def _compose_regularizations():
    regs = [decorrelation_reg]
    reg_grad = jax.grad(lambda x: sum([1.0, ] + [reg(x) for reg in regs]))
    return jax.jit(reg_grad)

grad_reg = _compose_regularizations()
print(f"{grad_reg=}")
```

    grad_reg=<PjitFunction of <function _compose_regularizations.<locals>.<lambda> at 0x0000026B186B7380>>
    

### Пересчет Фи на шаге


```python
# расчет нового значения Фи по результатам шага
def _calc_phi(
        *,
        batch: jax.Array,
        phi: jax.Array,
        p_ti: jax.Array,
        grad_reg,
    ):
    # jnp.add.at - jax.numpy.ufunc.at(a, indices[, b, inplace])
    # применяет функцию ufunc к элементам a по индексам indices, b - аргумент, inplace=True имитирует по_месту 
    # здесь создает новый массив на основе нулевого массива формы phi и для токенов, полученных в батче, прибавляет p_ti
    # прибавляет столько раз, сколько встречается в батче
    # в результате матрица размерностью как Фи, но ненормализованная - суммы вероятностей
    # дальше применяем регуляризаторы и нормализуем
    phi_new = jnp.add.at(
        jnp.zeros_like(phi),
        batch,
        p_ti,
        inplace=False,
    )  # (W, T)
    print(f"{phi_new[0]=}")
    phi_new -= phi * grad_reg(phi)  # (W, T)
    print(f"{phi_new[0]=}")
    phi_new = _norm(phi_new)  # (W, T)
    print(f"{phi_new[0]=}")
    return phi_new

phi_new = _calc_phi( 
            batch=batch,
            phi=phi,
            p_ti=p_ti,
            grad_reg=grad_reg)

```

    phi_new[0]=Array([0.01548281, 0.07350703, 0.09873346, 0.14708544, 0.0881432 ,
           0.02225126, 0.16040105, 0.03219692, 0.06452698, 0.29767174],      dtype=float32)
    phi_new[0]=Array([0.00068657, 0.03804044, 0.07118985, 0.11466126, 0.072062  ,
           0.01310876, 0.1280796 , 0.02339669, 0.04806345, 0.262158  ],      dtype=float32)
    phi_new[0]=array([0.00055184, 0.0339425 , 0.05590081, 0.07988738, 0.0408109 ,
           0.00743632, 0.07810109, 0.01676256, 0.02814333, 0.13296215],
          dtype=float32)
    

### Использование рассчитанных Фи и n_t в следующем шаге до сходимости
