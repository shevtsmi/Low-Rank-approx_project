#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
import os
import pandas
os.getcwd()

#from func.Givens_Rotation import formrot, Givens_left, Givens_right
from func.Tridiag_Rotations import Tridiagonal_chase

# A = 1.0 * np.array([
#     [1, 2, 3,  4,   5, 6,  19],
#     [0, 9, 13, 0,   0, 0,  0],
#     [0, 0, 8,  12,  0, 0,  0],
#     [0, 0, 0,   7, 11, 0,  0],
#     [0, 0, 0,   0,  6, 10, 0],
#     [0, 0, 0,   0,  0, 5,  66], 
#     [0, 0, 0,   0,  0, 0,  73], 
#     ])

A = np.zeros((7, 7)) + np.diag([1, -1, 1, 0, 0, 1, 0])

# A = 1.0 * np.array([
#     [1, 2, 3, 4, 5],
#     [0, 9, 13, 0, 0],
#     [0, 0, 8, 12, 0],
#     [0, 0, 0, 7, 11],
#     [0, 0, 0, 0, 6]

#     ])

Q, B, W = Tridiagonal_chase(A) 

print("A\n", A)


#np.savetxt("array.txt", np.matrix(Q.dot(A.dot(W))), fmt="%s")

print("\nFinal output (upper tridiagonal matrix) B : \n")
df = pandas.DataFrame(B)
print(df)
print("\n")
print("Видим, что матрицы Q, W также посчитаны правильно, т.е. QAW - B = 0 : \n")
print(pandas.DataFrame(Q.dot(A.dot(W)) - B))
