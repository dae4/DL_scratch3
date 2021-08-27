#%%
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dezero.core import as_variable
import numpy as np
from dezero.models import MLP
import dezero.functions as F
from dezero import Variable
from dezero import optimizers

x = Variable(np.array([[1,2,3],[4,5,6]]))
y = F.get_item(x,1)
print(y)

# %%
y.backward()
print(x.grad)
# %%
x = Variable(np.array([[1,2,3],[4,5,6]]))
indices = np.array([0,0,1])
y = F.get_item(x,indices)
print(y)

# %%

Variable.__getitem__ = F.get_item
y = x[1]
print(y)

y = x[:2]
print(y)
# %%
from dezero.models import MLP

model = MLP((10,3))
# %%
x = np.array([[0.2,-0.4]])
y = model(x)
print(y)

#%%
def softmax1d(x):
    x = as_variable(x)
    y = F.exp(x)
    sum_y = F.sum(y)
    return y / sum_y
# %%
