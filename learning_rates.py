from itertools import count
import numpy as np


def exponential_lr(eta, gamma):
    if gamma < 0:
        raise ValueError('Gamma must be positive or equal to 0')

    for i in count():
        yield eta * gamma ** (i + 1)


def linear_lr(a, b, max_iter):
    if a > b:
        raise ValueError('Right limit must be greater than left limit')

    for i in range(max_iter):
        yield b - (b - a) * i / max_iter


def cosine_lr(a, b, max_iter):
    if a > b:
        raise ValueError('Right limit must be greater than left limit')

    for i in range(max_iter):
        yield a + 0.5 * (b - a) * (1 + np.cos(np.pi * i / max_iter))


def inv_scale_lr(step_alpha=1, step_beta=0.0):
    if step_alpha < 0 or step_beta < 0:
        raise ValueError('alpha and beta can not be below zero')

    for i in count():
        yield step_alpha / ((i + 1) ** step_beta)
