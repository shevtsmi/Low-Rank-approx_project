#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import math
import os
os.getcwd()


def sign(a):
    if(a == 0):
        return 0
    if(a > 0):
        return 1
    return -1


def formrot(a, b):
    c = 0
    s = 0
    if(a == 0 and b == 0):
        #print("Error with a,b in formrot")
        return 0, 0
    if(b == 0):
        c = a
        s = 0
    else:
        if(math.fabs(a) <= math.fabs(b)):
            r = math.fabs(a / b)
            factor = math.sqrt(1 + r**2)
            c = sign(a) * r / factor
            s = - sign(b) / factor
        else:
            r = math.fabs(b / a)
            factor = math.sqrt(1 + r**2)
            c = sign(a) / factor
            s = - sign(b) * r / factor 
    return c, s


# i < j <= size - 1   -- индексы для ротации 
# c, s -- параметры, задающие ротацию
def Givens_left(c, s, size, i, j):
    G = np.eye(size, size)
    G[i][i] = c
    G[j][j] = c
    G[i][j] = -s
    G[j][i] = s
    return G

def Givens_right(c, s, size, i, j):
    G = np.eye(size, size)
    G[i][i] = c
    G[j][j] = c
    G[i][j] = s
    G[j][i] = -s
    return G