#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from Algorithm_2 import Algorithm_2
from Algorithm_3 import Algorithm_3
from func.Select_Index import Select_Columns, Select_Rows, Select_Ind

import itertools 
from itertools import combinations, chain 

def findsubsets(r, n): 
    s = range(0, n)
    return list(itertools.combinations(s, r)) 


#print(findsubsets())

def Check_Best_CUR(A, k):
    
    ans = math.inf
    columns = []
    rows = []
    
    m = A.shape[0]
    n = A.shape[1]
    
    Cols = findsubsets(k, n)
    Rows = findsubsets(k, m)
    
    for col__ in Cols:
        for row__ in Rows:
            
            col = list(col__)
            row = list(row__)
            
            C  = Select_Columns(A, col)
            R  = Select_Rows(A, row)
            U = np.linalg.pinv(C) @ A @ np.linalg.pinv(R)
            
            norm = np.linalg.norm(A - C @ U @ R)
            if(norm < ans):
                ans = norm
                columns = col
                rows = row
            
    return ans, columns, rows
            
m = 60
n = 80
k = 20

A = np.random.rand(m, n) * 10

#print("A = \n", A)

C, U, R, our_col, our_row = Algorithm_3(A, k) 

our_ans = np.linalg.norm(A - C @ U @ R) 
#true_ans, true_col, true_row = Check_Best_CUR(A, k)


# print("our answer : {} \ntrue answer : {}\n".format(our_ans, true_ans))

# print("True sets: ")
# print("Cols :\n {}\n".format(true_col))
# print("Rows :\n {}\n".format(true_row))

print("Our sets: ")
print("Cols :\n {}\n".format(our_col))
print("Rows :\n {}\n".format(our_row))







