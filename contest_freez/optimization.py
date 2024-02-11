import numpy as np
import time
from oracles import BinaryLogistic
from collections import Counter


class GDClassifier:
    """
    Реализация метода градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __optein_oracle(self, **kwargs):
        if self.loss_function == 'binary_logistic':
            self._oracle = BinaryLogistic(**kwargs)

    def __init__(
        # классические для задания
        self, loss_function, step_alpha=1, step_beta=0, tolerance=1e-5, max_iter=1000,
        warm_start=False, **kwargs  # дополнительные
    ):
        """
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия

        step_alpha - float, параметр выбора шага из текста задания

        step_beta- float, параметр выбора шага из текста задания

        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию.
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход

        max_iter - максимальное число итераций

        ---------

        warm_start - аргумент, специализирующий, нужно ли нам перезаписывать данные при повторном fit

        **kwargs - аргументы, необходимые для инициализации
        """
        self.loss_function = loss_function
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.warm_start = warm_start

        self.coefs = None

        self.__optein_oracle(**kwargs)

    def fit(self, X, y, w_0=None, trace=False):
        """
        Обучение метода по выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w_0 - начальное приближение в методе

        trace - переменная типа bool

        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Длина словаря history = количество итераций + 1 (начальное приближение)

        history['time']: list of floats, содержит интервалы времени между двумя итерациями метода
        history['func']: list of floats, содержит значения функции на каждой итерации
        (0 для самой первой точки)
        """

        BREAK_ON_MAX_ITER = True

        if w_0 is not None:
            self.coefs = w_0.copy()
        elif not self.warm_start or self.coefs is None:
            self.coefs = np.random.normal(size=X.shape[1])

        history = {
            'time': [time.time()],
            'func': [self._oracle.func(X, y, self.coefs)]
        }

        def stepper(k): return self.step_alpha / (k ** self.step_beta)
        for step in map(stepper, range(1, self.max_iter + 1)):
            self.coefs -= step * self._oracle.grad(X, y, self.coefs)

            history['time'].append(time.time())
            history['func'].append(self._oracle.func(X, y, self.coefs))

            if len(history['func']) > 1 and abs(history['func'][-1] - history['func'][-2]) < self.tolerance:
                BREAK_ON_MAX_ITER = False
                break

        # if BREAK_ON_MAX_ITER:
        #     print(
        #         'Model did not converged. Try increasing the max_iter attribute or make regularization stronger')

        return history if trace else self

    def predict(self, X):
        """
        Получение меток ответов на выборке X

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        return: одномерный numpy array с предсказаниями
        """

        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1) * 2 - 1

    def predict_proba(self, X):
        """
        Получение вероятностей принадлежности X к классу k

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        return: двумерной numpy array, [i, k] значение соответветствует вероятности
        принадлежности i-го объекта к классу k
        """
        if self.loss_function == 'binary_logistic':
            positive_proba = self._oracle.decision_function(X, self.coefs)

            return np.vstack([1 - positive_proba, positive_proba])

        return None

    def get_objective(self, X, y):
        """
        Получение значения целевой функции на выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array

        return: float
        """
        return self._oracle.func(X, y, self.coefs)

    def get_gradient(self, X, y):
        """
        Получение значения градиента функции на выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array

        return: numpy array, размерность зависит от задачи
        """

        return self._oracle.grad(X, y, self.coefs)

    def get_weights(self):
        """
        Получение значения весов функционала
        """
        return self.coefs


class SGDClassifier(GDClassifier):
    """
    Реализация метода стохастического градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(
        self, loss_function, batch_size=None, step_alpha=1, step_beta=0,
        tolerance=1e-5, max_iter=1000, random_seed=153, warm_start=False, **kwargs
    ):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия

        batch_size - размер подвыборки, по которой считается градиент
                     если batch_size - int => берутся наборы объектов batch_size размерности
                     если batch_size - float => берутся наборы объектов размерности мощности выборки на batch_size
                                                batch_size обязан быть в пределах 0 и 1

        step_alpha - float, параметр выбора шага из текста задания

        step_beta - float, параметр выбора шага из текста задания

        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход

        max_iter - максимальное число итераций (эпох)

        random_seed - в начале метода fit необходимо вызвать np.random.seed(random_seed).
        Этот параметр нужен для воспроизводимости результатов на разных машинах.

        ---------

        warm_start - аргумент, специализирующий, нужно ли нам перезаписывать данные при повторном fit

        **kwargs - аргументы, необходимые для инициализации
        """
        super().__init__(loss_function, step_alpha, step_beta,
                         tolerance, max_iter, warm_start, **kwargs)
        self.batch_size = 0.1 if batch_size is None else batch_size
        self.random_seed = random_seed

    def fit(self, X, y, w_0=None, trace=False, log_freq=1):
        """
        Обучение метода по выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w_0 - начальное приближение в методе

        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Если обновлять history после каждой итерации, метод перестанет
        превосходить в скорости метод GD. Поэтому, необходимо обновлять историю метода лишь
        после некоторого числа обработанных объектов в зависимости от приближённого номера эпохи.
        Приближённый номер эпохи:
            {количество объектов, обработанных методом SGD} / {количество объектов в выборке}

        log_freq - float от 0 до 1, параметр, отвечающий за частоту обновления.
        Обновление должно проиходить каждый раз, когда разница между двумя значениями приближённого номера эпохи
        будет превосходить log_freq.

        history['epoch_num']: list of floats, в каждом элементе списка будет записан приближённый номер эпохи:
        history['time']: list of floats, содержит интервалы времени между двумя соседними замерами
        history['func']: list of floats, содержит значения функции после текущего приближённого номера эпохи
        history['weights_diff']: list of floats, содержит квадрат нормы разности векторов весов с соседних замеров
        (0 для самой первой точки)
        """
        saved_random_state = np.random.get_state()
        np.random.seed(self.random_seed)

        BREAK_ON_MAX_ITER = True

        if w_0 is not None:
            self.coefs = w_0.copy()
        elif not self.warm_start or self.coefs is None:
            self.coefs = np.random.normal(size=X.shape[1])

        X_working = X.copy()
        batch_divider = self.batch_size if isinstance(
            self.batch_size, int) else int(X_working.shape[0] * self.batch_size)

        history = {
            'epoch_num': [0],
            'time': [time.time()],
            'func': [self._oracle.func(X, y, self.coefs)],
            'weights_diff': [np.linalg.norm(self.coefs) ** 2]
        }
        prev_coefs = self.coefs.copy()

        def stepper():
            i = 0
            while True:
                i += 1
                yield self.step_alpha / (i ** self.step_beta)

        step = stepper()
        for epoch_num in range(self.max_iter):
            proccessed_entities_count = 0
            np.random.shuffle(X_working)
            for X_batch in np.split(X_working, [*range(batch_divider, X_working.shape[0], batch_divider)]):
                proccessed_entities_count += X_batch.shape[0]
                self.coefs -= next(step) * \
                    self._oracle.grad(X, y, self.coefs)

                if proccessed_entities_count / X_working.shape[0] >= log_freq:
                    proccessed_entities_count = 0
                    history['time'].append(time.time())
                    history['func'].append(self._oracle.func(X, y, self.coefs))
                    history['epoch_num'].append(epoch_num)
                    history['weights_diff'].append(
                        np.linalg.norm(self.coefs - prev_coefs) ** 2)
                    prev_coefs = self.coefs.copy()

                if len(history['func']) > 1 and abs(history['func'][-1] - history['func'][-2]) < self.tolerance:
                    BREAK_ON_MAX_ITER = False
                    break

        np.random.set_state(saved_random_state)

        return history if trace else self
