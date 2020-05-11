#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import numpy as np
import os
os.getcwd()

from func.Select_Index import Select_Columns, Select_Rows, Select_Ind


n = 4      # размерность задачи

m = 5
n = 4
A = np.random.rand(m, n) * 10
S = [1, 3]
W = [0, 2]
print("A = \n {}".format(A))
print("=============\n")
print("S : {} \n\nW : {} \n\n A[:, S] = \n {} \n\n A[W, :] = \n {} \n\n A[W, S] = \n {} \n"\
      .format(S, W,  Select_Columns(A, S), Select_Rows(A, W), Select_Ind(A, W, S)))

