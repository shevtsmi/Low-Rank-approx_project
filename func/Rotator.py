#!/usr/bin/env python3
# -*- coding: utf-8 -*-



class Rotator():
    def __init__(self) :
        self.Q = []
        self.W = []
        self.q_size = 0
        self.w_size = 0
        
    def update_W(self, c, s, i, j):
        self.W.append([c, s, i, j])
        self.w_size += 1
        
    def update_Q(self, c, s, i, j):
        self.Q.append([c, s, i, j])
        self.q_size += 1
        
    def display(self):
        print("W : \n{}\nQ : \n{}\n".format(self.W, self.Q))
    
    
    def apply_W_T(self, A):
        
        for k in range(self.w_size):
            c = self.W[k][0]
            s = self.W[k][1]
            i = self.W[k][2]
            j = self.W[k][3]
            
            temp_i = A[i, :].copy()
            temp_j = A[j, :].copy()
            
            A[i, :] = c * temp_i - s * temp_j
            A[j, :] = s * temp_i + c * temp_j
        
        return A  
    
    def apply_Q_T(self, A):
        
        for k in range(self.q_size):
            c = self.Q[k][0]
            s = self.Q[k][1]
            i = self.Q[k][2]
            j = self.Q[k][3]

            temp_i = A[:, i].copy()
            temp_j = A[:, j].copy()
            
            A[:, i] = c * temp_i - s * temp_j
            A[:, j] = s * temp_i + c * temp_j
        return A
    
    
    
    
    def apply_W(self, A):
        
        for k in range(self.w_size):
            c = self.W[k][0]
            s = self.W[k][1]
            i = self.W[k][2]
            j = self.W[k][3]

            temp_i = A[:, i].copy()
            temp_j = A[:, j].copy()
            
            A[:, i] = c * temp_i - s * temp_j
            A[:, j] = s * temp_i + c * temp_j
        return A
    
    def apply_Q(self, A):
        for k in range(self.q_size):
            c = self.Q[k][0]
            s = self.Q[k][1]
            i = self.Q[k][2]
            j = self.Q[k][3]
            
            temp_i = A[i, :].copy()
            temp_j = A[j, :].copy()

            A[i, :] = c * temp_i - s * temp_j
            A[j, :] = s * temp_i + c * temp_j
        
        return A  
    
    def apply(self, A):
        A = self.apply_W(A)
        A = self.apply_Q(A)
        return A
    
    def apply_T(self, A):
        A = self.apply_W_T(A)
        A = self.apply_Q_T(A)
        return A