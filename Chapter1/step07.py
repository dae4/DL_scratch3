#%%
import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

class Function:
    def  __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)
        self.input = input
        self.output = output
        return output

class Square(Function):
    def forward(self,x):
        y = x**2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self,x):
        y = np.exp(x)
        return y

    def backward(self,gy):
        x = self.input.data
        gx = np.exp(x)*gy
        return gx


#%%
A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))

a = A(x)
b = B(a)
y = C(b)

## assert 문은 평가결과가 ture 가 아니면 예외가 발생함
## assert 조건

assert y.creator == C
assert y.creator.input == b
assert y.creator.input.creator == B
assert y.creator.input.creator.input == a
assert y.creator.input.creator.input.creator == A
assert y.creator.input.creator.input.creator.input == x
# %%
