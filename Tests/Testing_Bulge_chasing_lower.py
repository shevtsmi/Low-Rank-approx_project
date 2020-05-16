#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import numpy as np
import math
import pandas
import os
os.getcwd()

from func.Bulge_chasing_lower import Bulge_chasing_lower
from func.Bulge_chasing_upper import Bulge_chasing_upper


# Проверка:


n = 7      # размерность задачи

s = np.zeros(n)   # вектор [0 ... (n - 1)], задающий диагональную матрицу Sigma

x = np.zeros(n)   # вектор, относительно которого происходит нормировка

# Для примера положим:
for i in range(n):
    s[i] = i + 1.0
    x[i] = i + 1



# # проверка поведения с нулевыми столбцами
# s[n - 1] = 0
# s[n - 2] = 0


Sigma = np.diag(s)

Q, W = Bulge_chasing_lower(s, x, n)

print(pandas.DataFrame(Sigma))
print("=================")
print(pandas.DataFrame(Q.dot(Sigma.dot(W)))   )     # смотрим на вывод, убеждаемся, что верхняя треугольная
print("\n")
print("Exact norm(z) = {} \n".format(np.linalg.norm(x)))     # это настоящая норма вектора x

W_T = W.T
print("Computed:", np.linalg.norm(W_T.dot(x)))               # это || z || * e_1

# как видим они равны, алгоритм работает верно (можно увеличить n, но вывод будет труднее читаем)
