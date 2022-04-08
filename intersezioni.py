#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 17:27:09 2022

@author: defo
"""
import math
import numpy as np
import matplotlib.pyplot as plt

xg=1
yg=1

n=5

fig,ax = plt.subplots()
x=np.arange(-10,10,0.01)
alpha=0
for i in range(5):
    print(i)
    alpha+=((2*math.pi)/n)
    if alpha!=math.pi:
        print((alpha*360)/(2*math.pi))
        m=math.tan(alpha)
        ax.plot(x,m*(x-xg)+yg,label="{coeff}".format(coeff=m))
    else :
        print("ciao")




ax.set_aspect('equal', 'box')
ax.ylim(0,10)


plt.show()