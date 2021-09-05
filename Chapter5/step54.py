#%%
import numpy as np

dropout_ratio = 0.6
x = np.ones(10)

mask = np.random.rand(10) > dropout_ratio
y = x * mask
# -- Direct Dropout --
## for train
mask = np.random.rand(*x.shape) > dropout_ratio
y = x * mask

## for test
scale = 1 - dropout_ratio
y = x* scale

# -- Inverted Dropout -- 

## from train
scale = 1 - dropout_ratio
mask = np.random.rand(*x.shape)
y = x * mask / scale

## for test
y = x
#%%
#%%
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import test_mode
import dezero.functions as F

x = np.ones(5)
print(x)

y = F.dropout(x)
print(y)

with test_mode():
    y = F.dropout(x)
    print(y)
# %%
