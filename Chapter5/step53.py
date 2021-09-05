#%%
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dezero.core import Parameter
import numpy as np

x = np.array([1,2,3])
np.save('test.npy',x)

x = np.load('test.npy')
print(x)
# %%
x1 = np.array([1,2,3])
x2 = np.array([4,5,6])

np.savez('test.npz',x1=x1,x2=x2)

arrays = np.load('test.npz')
x1 = arrays['x1']
x2 = arrays['x2']
print(dir(arrays))
print(x1)
print(x2)
# %%
x1 = np.array([1,2,3])
x2 = np.array([4,5,6])
data={'x1':x1, 'x2':x2}

np.savez('test.npz',**data)

arrays = np.load('test.npz')
x1 = arrays['x1']
x2 = arrays['x2']
print(x1)
print(x2)

# %%
from dezero.layers import Layer
layer = Layer()

l1 = Layer()
l1.p1 = Parameter(np.array(1))

layer.l1 = l1
layer.p2 = Parameter(np.array(2))
layer.p3 = Parameter(np.array(3))

# %%

params_dict={}
layer._flatten_params(params_dict)
print(params_dict)
# %%
import os
import dezero
import dezero.functions as F
from dezero import optimizers
from dezero import DataLoader
from dezero.models import MLP

max_epoch = 3
batch_size = 100

train_set = dezero.datasets.MNIST(train=True)
train_loader = DataLoader(train_set, batch_size)
model = MLP((1000, 10))
optimizer = optimizers.SGD().setup(model)

if os.path.exists('my_mlp.npz'):
    model.load_weights('my_mlp.npz')



for epoch in range(max_epoch):
    sum_loss = 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(t)

    print('epoch: {}, loss: {:.4f}'.format(epoch + 1, sum_loss / len(train_set)))

model.save_weights('my_mlp.npz')
# %%
