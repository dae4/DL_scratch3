#%%
if '__file__' in globals(): ## search dezero 
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__),".."))

import numpy as np 
import dezero.functions as F
from dezero import Variable

a = np.array([1,2,3])
b = np.array([4,5,6])
c= np.dot(a,b)
print(c)

a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])
c= np.dot(a,b)
print(c)


# %%
