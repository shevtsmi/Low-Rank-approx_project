#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
выбирает подматрицу по индексному множеству
Input: 
        A in R^{m x n} -- матрица
        S -- множество индексов [0; n - 1]
Output: 
        Columns -- A(:, S)  (выбирает столбцы)
        Rows    -- A(:, S)  (выбирает строки)
        Ind     -- A(W, S)  (выбирает подматрицу по индексам строк W и столбцов S)
"""


import numpy as np



def Select_Columns(A, S):
    m = A.shape[0]
    S.sort()
    size = len(S)
    if(size > m or size == 0):
        print("Error in Select_Columns : columns requested : {}, \
              but matrix has only {}".format(size, m))
    Ans = np.zeros((m, size))
    for i in range(size):
        Ans[:, i] = A[:, S[i]]
    return Ans

def Select_Rows(A, S):
    n = A.shape[1]
    S.sort()
    size = len(S)
    if(size > n or size == 0):
        print("Error in Select_Rows : rows requested : {}, \
              but matrix has only {}".format(size, n))
    Ans = np.zeros((size, n))
    for i in range(size):
        Ans[i, :] = A[S[i], :]
    return Ans

def Select_Ind(A, Rows, Columns):
    m = A.shape[0]
    n = A.shape[1]
    
    size_m = len(Rows)
    size_n = len(Columns)
    
    if(size_m > m or size_n > n\
       or size_m == 0 or size_n == 0):
        print("Error in Select_Rows : columns/rows requested : {}x{}, \
              but matrix has only {}x{}".format(size_m, size_n, m, n))  
    return Select_Columns(Select_Rows(A, Rows), Columns)
        



















