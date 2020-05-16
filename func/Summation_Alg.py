#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import os
os.getcwd()

def GetCoFromEigen(eigen):
    n = len(eigen)
    ans = []
    S = np.zeros((n, n+1), dtype = np.complex)
    S[0,1] = eigen[0]
    for i in range(n-1):
        S[i, 0] = 1
    for i in range(1, n):
        for j in range(1, i+2):
            S[i, j] = S[i-1, j] + eigen[i]*S[i-1,j-1]
    ans.append(complex(1.0,0))
    for i in range(1,n+1):
        ans.append(S[n-1, i])
    return ans


