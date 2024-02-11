from typing import Union
import numpy as np
from scipy import sparse, special


class BaseSmoothOracle:
    """
    Базовый класс для реализации оракулов.
    """

    @staticmethod
    def _get_dot_products(x: Union[np.ndarray, sparse.csr_matrix], y: np.ndarray, intercept: float = 0.0):
        return np.sum(x * y[None, :], axis=1) + intercept if isinstance(x, np.ndarray) else np.sum(x.multiply(y), axis=1) + intercept

    def func(self, w):
        """
        Вычислить значение функции в точке w.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, w):
        """
        Вычислить значение градиента функции в точке w.
        """
        raise NotImplementedError('Grad oracle is not implemented.')


class BinaryLogistic(BaseSmoothOracle):
    """
    Оракул для задачи двухклассовой логистической регрессии.

    Оракул должен поддерживать l2 регуляризацию.
    """

    def __init__(self, l2_coef):
        """
        Задание параметров оракула.

        l2_coef - коэффициент l2 регуляризации

        """

        self._l2_coef = l2_coef

    def func(self, X, y, w, intercept=None):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w - одномерный numpy array
        """
        margins = (super()._get_dot_products(X, w, intercept if intercept is not None else 0)).flat * y
        loss = np.mean(np.logaddexp(-margins, 0))
        regul = np.sum(w * w) * self._l2_coef / 2

        return loss + regul

    def grad(self, X, y, w, intercept=None):
        """
        Вычислить градиент функционала в точке w на выборке X с ответами y.

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w - одномерный numpy array
        """
        margins = (super()._get_dot_products(X, w, intercept if intercept is not None else 0)).flat * y
        grad_main = -np.mean(X.multiply(((1 - special.expit(margins)) * y)[:, None]), axis=0).A1 if not isinstance(X, np.ndarray) else \
                    -np.mean(X * ((1 - special.expit(margins)) * y)[:, None], axis=0)
        grad_regul = self._l2_coef * w
        grad_intercept = -np.mean((1 - special.expit(margins)) * y) if intercept is not None else None

        return grad_main + grad_regul, grad_intercept

    def decision_function(self, X, w, intercept=None):
        margins = super()._get_dot_products(X, w, intercept if intercept is not None else 0)
        return special.expit(margins)


class BinaryLogisticL1(BaseSmoothOracle):
    """
    Оракул для задачи двухклассовой логистической регрессии.

    Оракул должен поддерживать L1 регуляризацию.
    """

    def __init__(self, l1_coef):
        """
        Задание параметров оракула.

        l1_coef - коэффициент l1 регуляризации
        """

        self._l1_coef = l1_coef

    def func(self, X, y, w, intercept=None):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w - одномерный numpy array
        """

        margins = (super()._get_dot_products(X, w, intercept if intercept is not None else 0)).flat * y

        loss = np.mean(np.logaddexp(-margins, 0))
        regul = np.sum(np.abs(w) * self._l1_coef / 2)

        return loss + regul

    def grad(self, X, y, w, intercept=None):
        """
        Вычислить градиент функционала в точке w на выборке X с ответами y.

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w - одномерный numpy array
        """

        margins = (super()._get_dot_products(X, w, intercept if intercept is not None else 0)).flat * y
        grad_main = -np.mean(X.multiply(((1 - special.expit(margins)) * y)[:, None]), axis=0).A1 if not isinstance(X, np.ndarray) else \
                    -np.mean(X * ((1 - special.expit(margins)) * y)[:, None], axis=0)

        grad_regul = (self._l1_coef / 2) * np.sign(w)

        grad_intercept = -np.mean(special.expit(-margins) * y) if intercept is not None else None

        return grad_main + grad_regul, grad_intercept

    def decision_function(self, X, w, intercept=None):
        margins = super()._get_dot_products(X, w, intercept if intercept is not None else 0)
        return special.expit(margins)
