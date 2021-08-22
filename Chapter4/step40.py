#%%
if '__file__' in globals(): ## search dezero 
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__),".."))

import numpy as np 
import dezero.functions as F
from dezero import Variable

x = np.array([1,2,3])
y = np.broadcast_to(x,(2,3))
print(y)
# %%
from dezero.utils import sum_to

x = np.array([[1,2,3],[4,5,6]])
y = sum_to(x,(1,3))
print(y)

y = sum_to(x,(2,1))
print(y)
# %%
x0 = np.array([1,2,3])
x1 = np.array([10])
y = x0+x1
print(y)
# %%
x0 = Variable(np.array([1,2,3]))
x1 = Variable(np.array([10]))
y = x0+x1
print(y)

# %%
x0 = Variable(np.array([1,2,3]))
x1 = Variable(np.array([10]))
y = x0+x1
print(y)
y.backward()
print(x1.grad)
