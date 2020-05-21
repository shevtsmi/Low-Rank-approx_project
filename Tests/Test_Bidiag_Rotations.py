#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import math
import os,sys,inspect


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from func.Bidiag_Rotations import house_bidiag

n = 6
m = 6
A = np.random.rand(m, n) * 10

#A = np.arange(1,13.0).reshape(4,3)
A_test = np.copy(A)
print(A)

U, B, Vt = house_bidiag(A, full=True)
print("B:\n", B)
print("U:\n", U)
print("Vt:\n", Vt)

print("should equal input:", np.allclose(U.dot(B).dot(Vt), A_test))


B = house_bidiag(A, full=False)
print("B:\n", B)





