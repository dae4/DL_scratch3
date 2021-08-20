#%%
if '__file__' in globals(): ## search dezero 
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__),".."))

import numpy as np 
import dezero.functions as F
from dezero import Variable
x = np.array([[1,2,3],[4,5,6]])
y = np.reshape(x,(6,))
print(y)
# %%
x = Variable(np.array([[1,2,3],[4,5,6]]))
y = F.reshape(x,(6,))
y.backward(retain_grad=True)
print(x.grad)
# %%

x = np.random.rand(1,2,3)
y = x.reshape((2,3))
print(y)
y = x.reshape([2,3])
print(y)
y = x.reshape(2,3)
print(y)
# %%
x = Variable(np.random.rand(1,2,3))
y = x.reshape((2,3))
y = x.reshape(2,3)
print(y)

# %%
x = np.array([[1,2,3],[4,5,6]])
y = np.transpose(x)
print(y)
# %%
x = Variable(np.array([[1,2,3],[4,5,6]]))
y = F.transpose(x)
y.backward()
print(x.grad)

# %%
x = Variable(np.random.rand(2,3))
y = x.transpose()
y = x.T
print(y)
# %%
