import numpy as np
import time

from oracles import *
from learning_rates import *


class GDClassifier:
    """
    Реализация метода градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __optein_oracle(self, **kwargs):
        if self.loss_function == 'binary_logistic':
            self._oracle = BinaryLogistic(self.l2_coef, **kwargs)
        if self.loss_function == 'binary_logistic_L1':
            self._oracle = BinaryLogisticL1(self.l1_coef, **kwargs)

    def __init__(

        self, loss_function='binary_logistic', *, learning_rate=inv_scale_lr(), tolerance=1e-5, max_iter=100,
        warm_start=False, fit_intercept=True, l2_coef=0, l1_coef=0, **kwargs
    ):
        """
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия

        learning_rate - функция learning_rate, что дает шаги градиента. Определена в модуле learning_rate

        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию.
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход

        max_iter - максимальное число итераций

        warm_start - аргумент, специализирующий, нужно ли нам перезаписывать данные при повторном fit

        fit_intercept - bool, определяет, хотим ли мы добавлять свободный коэффициент

        **kwargs - аргументы, необходимые для инициализации
        """
        self.loss_function = loss_function
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.warm_start = warm_start
        self.learning_rate = learning_rate
        self.fit_intercept = fit_intercept
        self.l2_coef = l2_coef
        self.l1_coef = l1_coef

        self.coefs = None
        self.intercept = None

        self.__optein_oracle(**kwargs)

    def fit(self, X, y, w_0=None, intercept=None, trace=False, X_test=None, y_test=None):
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

        X_test - если не None, то если trace=True и y_test не None => будет считаться precision

        y_test - если не None, то если trace=True и X_test не None => будет считаться precision

        """
        if (X_test is not None and y_test is None) or (X_test is None and y_test is not None):
            raise ValueError('X_train and y_train both must be None or not None')

        BREAK_ON_MAX_ITER = self.tolerance is not None

        if w_0 is not None:
            self.coefs = w_0.copy()
        elif not self.warm_start or self.coefs is None:
            self.coefs = np.random.normal(size=X.shape[1])

        if self.fit_intercept:
            if intercept is not None:
                self.intercept = intercept
            elif not self.warm_start or self.intercept is None:
                self.intercept = np.random.normal(size=(1))[0]

        history = {
            'time': [time.time()],
            'func': [self._oracle.func(X, y, self.coefs, self.intercept)],
            'precision': [],
            'precision_train': [],
            'recall': [],
            'recall_train': [],
            'accuracy': [],
            'accuracy_train': [],
        }
        for _ in range(self.max_iter):
            step = next(self.learning_rate)
            grad_body, grad_intercept = self._oracle.grad(X, y, self.coefs, self.intercept)
            self.coefs -= step * grad_body
            if self.fit_intercept:
                self.intercept -= step * grad_intercept

            history['time'].append(time.time())
            history['func'].append(self._oracle.func(X, y, self.coefs, self.intercept))

            if trace and X_test is not None:
                probs = self.predict_proba(X_test)
                positives = np.argmax(probs, axis=1).flat == 1
                negatives = np.argmax(probs, axis=1).flat == 0
                true_pos = positives & (y_test == 1)
                true_neg = negatives & (y_test == -1)
                history['precision'].append(np.sum(true_pos) / np.sum(positives) if np.sum(positives) != 0 else 0)
                history['recall'].append(np.sum(true_pos) / np.sum(y_test == 1))
                history['accuracy'].append((np.sum(true_pos) + np.sum(true_neg)) / X_test.shape[0])

                probs_train = self.predict_proba(X)
                positives_train = np.argmax(probs_train, axis=1).flat == 1
                negatives_train = np.argmax(probs_train, axis=1).flat == 0
                true_pos_train = positives_train & (y == 1)
                true_neg_train = negatives_train & (y == -1)
                history['precision_train'].append(np.sum(true_pos_train) / np.sum(positives_train) if np.sum(positives_train) != 0 else 0)
                history['recall_train'].append(np.sum(true_pos_train) / np.sum(y == 1))
                history['accuracy_train'].append((np.sum(true_pos_train) + np.sum(true_neg_train)) / X.shape[0])

            if self.tolerance is not None and \
               len(history['func']) > 1 and abs(history['func'][-1] - history['func'][-2]) < self.tolerance:
                BREAK_ON_MAX_ITER = False
                break

        if BREAK_ON_MAX_ITER:
            print('WARNING: Model did not converged. Try increasing the max_iter attribute or make regularization stronger', end='\n\n')
            history['max_iter_limit'] = True

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
        positive_proba = self._oracle.decision_function(X, self.coefs, self.intercept)

        return np.hstack([1 - positive_proba, positive_proba])

    def get_objective(self, X, y):
        """
        Получение значения целевой функции на выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array

        return: float
        """
        return self._oracle.func(X, y, self.coefs, self.intercept)

    def get_gradient(self, X, y):
        """
        Получение значения градиента функции на выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array

        return: numpy array, размерность зависит от задачи
        """
        return self._oracle.grad(X, y, self.coefs, self.intercept)

    def get_weights(self):
        """
        Получение значения весов функционала
        """
        return self.coefs

    def get_intercept(self):
        """
        Получить значение intercept
        """
        return self.intercept


class SGDClassifier(GDClassifier):
    """
    Реализация метода стохастического градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(
        self, loss_function='binary_logistic', *, batch_size=None, learning_rate=inv_scale_lr(), l2_coef=1, l1_coef=1,
        tolerance=1e-5, max_iter=100, random_seed=153, warm_start=False, fit_intercept=True, **kwargs
    ):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия

        batch_size - размер подвыборки, по которой считается градиент
                     если batch_size - int => берутся наборы объектов batch_size размерности
                     если batch_size - float => берутся наборы объектов размерности мощности выборки на batch_size
                                                batch_size обязан быть в пределах 0 и 1

        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход

        max_iter - максимальное число итераций (эпох)

        random_seed - в начале метода fit необходимо вызвать np.random.seed(random_seed).
        Этот параметр нужен для воспроизводимости результатов на разных машинах.

        ---------

        learning_rate - функция learning_rate, что дает шаги градиента. Определена в модуле learning_rate

        warm_start - аргумент, специализирующий, нужно ли нам перезаписывать данные при повторном fit

        fit_intercept - bool, определяет, хотим ли мы добавлять свободный коэффициент


        **kwargs - аргументы, необходимые для инициализации
        """
        super().__init__(loss_function=loss_function, learning_rate=learning_rate, tolerance=tolerance, max_iter=max_iter,
                         warm_start=warm_start, fit_intercept=fit_intercept,
                         l2_coef=l2_coef, l1_coef=l1_coef, **kwargs)

        self.batch_size = 0.1 if batch_size is None else batch_size
        self.random_seed = random_seed

    def fit(self, X, y, w_0=None, intercept=None, trace=False, log_freq=1, X_test=None, y_test=None):
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

        if (X_test is not None and y_test is None) or (X_test is None and y_test is not None):
            raise ValueError('X_train and y_train both must be None or not None')

        BREAK_ON_MAX_ITER = self.tolerance is not None

        if w_0 is not None:
            self.coefs = w_0.copy()
        elif not self.warm_start:
            self.coefs = np.random.normal(size=X.shape[1])

        if self.fit_intercept:
            if intercept is not None:
                self.intercept = intercept
            elif not self.warm_start or self.intercept is None:
                self.intercept = np.random.normal(size=(1))[0]

        X_working = X.copy()
        y_working = y.copy()
        batch_divider = self.batch_size if isinstance(self.batch_size, int) else int(X_working.shape[0] * self.batch_size)

        history = {
            'epoch_num': [0],
            'time': [time.time()],
            'func': [self._oracle.func(X, y, self.coefs, self.intercept)],
            'weights_diff': [np.linalg.norm(self.coefs) ** 2],
            'precision': [],
            'precision_train': [],
            'recall': [],
            'recall_train': [],
            'accuracy': [],
            'accuracy_train': [],
        }
        prev_coefs = self.coefs.copy()

        for epoch_num in range(self.max_iter):
            proccessed_entities_count = X_working.shape[0] + 1
            perm = np.random.permutation(X_working.shape[0])
            X_working = X_working[perm]
            y_working = y_working[perm]

            for i in range(0, X_working.shape[0] - batch_divider + 1, batch_divider):
                step = next(self.learning_rate)
                X_batch = X_working[i:i + batch_divider]
                y_batch = y_working[i:i + batch_divider]
                proccessed_entities_count += X_batch.shape[0]
                grad_body, grad_intercept = self._oracle.grad(X_batch, y_batch, self.coefs, self.intercept)

                self.coefs -= step * grad_body
                if self.fit_intercept:
                    self.intercept -= step * grad_intercept

                if proccessed_entities_count / X_working.shape[0] >= log_freq:
                    proccessed_entities_count = 0
                    history['time'].append(time.time())
                    history['func'].append(self._oracle.func(X, y, self.coefs, self.intercept))
                    history['epoch_num'].append(epoch_num)
                    history['weights_diff'].append(
                        np.linalg.norm(self.coefs - prev_coefs) ** 2)
                    prev_coefs = self.coefs.copy()

                    if trace and X_test is not None:

                        probs = self.predict_proba(X_test)
                        positives = np.argmax(probs, axis=1).flat == 1
                        negatives = np.argmax(probs, axis=1).flat == 0
                        true_pos = positives & (y_test == 1)
                        true_neg = negatives & (y_test == -1)
                        history['precision'].append(np.sum(true_pos) / np.sum(positives) if np.sum(positives) != 0 else 0)
                        history['recall'].append(np.sum(true_pos) / np.sum(y_test == 1))
                        history['accuracy'].append((np.sum(true_pos) + np.sum(true_neg)) / X_test.shape[0])

                        probs_train = self.predict_proba(X)
                        positives_train = np.argmax(probs_train, axis=1).flat == 1
                        negatives_train = np.argmax(probs_train, axis=1).flat == 0
                        true_pos_train = positives_train & (y == 1)
                        true_neg_train = negatives_train & (y == -1)
                        history['precision_train'].append(np.sum(true_pos_train) / np.sum(positives_train) if np.sum(positives_train) != 0 else 0)
                        history['recall_train'].append(np.sum(true_pos_train) / np.sum(y == 1))
                        history['accuracy_train'].append((np.sum(true_pos_train) + np.sum(true_neg_train)) / X.shape[0])

            if self.tolerance is not None and \
                    len(history['func']) > 1 and abs(history['func'][-1] - history['func'][-2]) < self.tolerance:
                BREAK_ON_MAX_ITER = False
                break

        if BREAK_ON_MAX_ITER:
            print('WARNING: Model did not converged. Try increasing the max_iter attribute or make regularization stronger', end='\n\n')
            history['max_iter_limit'] = True

        np.random.set_state(saved_random_state)

        return history if trace else self
