#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
#import os
import pandas
#os.getcwd()
import os,sys,inspect


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

#from func.Givens_Rotation import formrot, Givens_left, Givens_right
from func.Tridiag_Rotations import Tridiagonal_chase

A = 1.0 * np.array([
    [1, 2, 3,  4,   5, 6,  19],
    [0, 9, 13, 0,   0, 0,  0],
    [0, 0, 8,  12,  0, 0,  0],
    [0, 0, 0,   7, 11, 0,  0],
    [0, 0, 0,   0,  6, 10, 0],
    [0, 0, 0,   0,  0, 5,  66], 
    [0, 0, 0,   0,  0, 0,  73], 
    ])

#A = np.zeros((7, 7)) + np.diag([1, -1, 1, 0, 0, 1, 0])

# A = 1.0 * np.array([
#     [1, 2, 3, 4, 5],
#     [0, 9, 13, 0, 0],
#     [0, 0, 8, 12, 0],
#     [0, 0, 0, 7, 11],
#     [0, 0, 0, 0, 6]

#     ])

B, Giv = Tridiagonal_chase(A, full=True) 

print("A\n", A)


#np.savetxt("array.txt", np.matrix(Q.dot(A.dot(W))), fmt="%s")

# print("\nGIV: \n")
# print(pandas.DataFrame(Giv.apply(A)))
df = pandas.DataFrame(B)
print("B = \n{}\n".format(df))
print("\n")
# print("Error QAW - B : \n")
# print(np.linalg.norm(Q.dot(A.dot(W)) - B))
print("Giv - B: \n")
print(np.linalg.norm(Giv.apply(A) - B))
#print("Error Rotator : {}\n".format(np.linalg.norm(Giv.apply(A) - B)))



B = Tridiagonal_chase(A, full=False) 

df = pandas.DataFrame(B)
print("B = \n{}\n".format(df))
print("\n")















