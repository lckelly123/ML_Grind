import math
import time
import numpy as np
import torch

#Main model: E[Y | X=x] can be expressed as weighted sum of features x
#yhat=w1x1+w2x2+...wdxd + b = wTx + b
#Entire dataset of n examples: yhat=Xw+b where broadcasting is used and X within(n,d)

#use squared error loss

#value of vectorization:
n=10000
a=torch.ones(n)
b=torch.ones(n)

c = torch.zeros(n)
t = time.time()
for i in range(n):
    c[i] = a[i] + b[i]
f'{time.time() - t:.5f} sec'

t = time.time()
d = a + b
f'{time.time() - t:.5f} sec'


