#%%
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import cupy as cp

x= cp.arrange(6).reshape(2,3)
print(x)

y = x.sum(axis=1)
print(y)
#%%
import numpy as np
import cupy as cp

n = np.array([1,2,3])
c = cp.asarray(n)
assert type(c) == cp.ndarray

c = cp.array([1,2,3])
n = cp.asnumpy(c)
assert type(n) == np.ndarray
#%%
x = np.array([1,2,3])
xp = cp.get_array_modelue(x)
assert xp == np

x = cp.array([1,2,3])
xp = cp.get_array_module(x)
assert xp == cp

#%%