#%%
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dezero.optimizers import Optimizer
import numpy as np
import dezero.layers as L

rnn = L.RNN(10)
x = np.random.rand(1,1)
h = rnn(x)
print(h.shape)
# %%
from dezero import Model 
import dezero.layers as L
import dezero.functions as F

class SimpleRNN(Model):
    def __init__(self, hidden_size , out_size):
        super().__init__()
        self.rnn = L.RNN(hidden_size)
        self.fc = L.Linear(out_size)

    def reset_state(self):
        self.rnn.reset_state()
    
    def forward(self, x):
        h = self.rnn(x)
        y = self.fc(h)
        return y
# %%
seq_data = [np.random.randn(1,1) for _ in range(1000)]
xs = seq_data[0:-1]
ts = seq_data[1:]

model = SimpleRNN(10,1)
loss, cnt = 0,0
for x, t in zip(xs,ts):
    y = model(x)
    loss += F.mean_squared_error(y,t)
    cnt += 1
    if cnt == 2:
        model.cleargrads()
        loss.backward()
        break
# %%
import numpy as np
import dezero
import matplotlib.pyplot as plt
import dezero.datasets

train_set = dezero.datasets.SinCurve(train=True)
print(len(train_set))
print(train_set[0])
print(train_set[1])
print(train_set[2])

xs = [example[0] for example in train_set]
ts = [example[0] for example in train_set]
plt.plot(np.arange(len(xs), xs, label='xs'))
plt.plot(np.arange(len(xs), ts, label='ts'))
plt.show()
# %%
max_epoch = 100
hidden_size = 100
bptt_length = 30
train_set = dezero.datasets.SinCurve(trian=True)
seqlen = len(train_set)

model = SimpleRNN(hidden_size, 1)
optimizer = dezero.optimizers.Adam().setup(model)

for epoch in range(max_epoch):
    model.reset_state()
    loss, count = 0,0 

    for x,t in train_set:
        x = x.reshape(1,1)
        y = model(x)
        loss += F.mean_squared_error(y,t)
        count += 1

        if count % bptt_length ==0 or count == seqlen:
            model.cleargrads()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()

        avg_loss = float(loss.data) / count 
        print('| epoch  %d | loss %f' % ( epoch + 1,avg_loss))
# %%
