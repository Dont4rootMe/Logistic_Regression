import numpy as np


def grad_finite_diff(function, w, eps=1e-8):
    """
    Возвращает численное значение градиента, подсчитанное по следующией формуле:
        result_i := (f(w + eps * e_i) - f(w)) / eps,
        где e_i - следующий вектор:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """

    anchor_value = function(w)
    eps_func = np.array([function(w + eps * e) for e in np.eye(w.shape[0])])

    return (eps_func - anchor_value) / eps
