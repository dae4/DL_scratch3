#%%
if '__file__' in globals(): ## search dezero 
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__),".."))

import numpy as np
from dezero import Function

class Sin(Function):
    def forward(self,x):
        y=np.sin(x)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy*np.cos(x)
        return gx
        
def sin(x):
    return Sin()(x)