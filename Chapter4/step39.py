#%%
if '__file__' in globals(): ## search dezero 
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__),".."))

import numpy as np 
import dezero.functions as F
from dezero import Variable

x = Variable(np.array([1.2,3,4,5,6]))
y = F.sum(x)
y.backward()
print(y)
print(x.grad)
# %%
x = Variable(np.array([[1,2,3],[4,5,6]]))
y = F.sum(x)
y.backward()
print(y)
print(x.grad)
# %%
x = np.array([[1,2,3],[4,5,6]])
y = np.sum(x, axis=0)
print(y)
print(x.shape,'->',y.shape)
# %%
x = np.array([[1,2,3],[4,5,6]])
y = np.sum(x, keepdims=True)
print(y)
print(y.shape)
# %%
x = Variable(np.array([[1,2,3],[4,5,6]]))
y = F.sum(x, axis=0)
y.backward()
print(y)
print(x.grad)

x = Variable(np.random.randn(2,3,4,5))
y = x.sum(keepdims=True)
print(y.shape)
# %%
