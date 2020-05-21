#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from func.Bulge_chasing_lower import Bulge_chasing_lower
from func.Rotator import Rotator

        
Giv = Rotator();
Giv.update_Q(1, 2, 0, 1);
Giv.update_W(2, 1, 0, 1);
Giv.update_W(3, 4, 0, 1);

A = np.array([[1.0, 0], 
     [0, 1.0]])
print(Giv.apply(A))























