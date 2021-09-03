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
import time
import dezero.functions as F
from dezero import optimizers
from dezero import DataLoader
from dezero.models import MLP
import dezero.cuda

max_epoch = 5 
batch_size = 100
lr = 1.0

train_set = dezero.datasets.MNIST(train=True)
train_loader = DataLoader(train_set,batch_size)
model = MLP((1000,10))
optimizer = optimizers.SGD(lr).setup(model)

if dezero.cuda.gpu_enable:
    train_loader.to_gpu()
    model.to_gpu()

for epoch in range(max_epoch):
    start = time.time()
    sum_loss = 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y,t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
    elapsed_time = time.time() - start
    print('epoch : {}'.format(epoch+1))
    print('train loss : {:4}, time{:4f}[sec]'.format(sum_loss/len(train_set),elapsed_time))


# %%
