#%%
from common import square,as_array,add
import weakref
import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{}은 지원하지 않음".format(type(data)))
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self,retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs=[]
        seen_set =set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x : x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs,tuple):
                gxs=(gxs,)
        
            for x, gx in zip(f.inputs,gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                    
                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

class Function:
    def  __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs) ## list unpack
        if not isinstance(ys,tuple):
            ys=(ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop: 
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for ouput in outputs]
            return outputs if len(outputs) > 1 else outputs[0]

    def forward(self,xs):
        raise NotImplementedError()
        
    def backward(self,gys):
        raise NotImplementedError()


class Config:
    enable_backprop = True


#%%
x0 = Variable(np.array(1.0))
x1 = Variable(np.array(1.0))
t = add(x0,x1)
y = add(x0,t)
y.backward()

print(y.grad, t.grad)
print(x0.grad, x1.grad)
# %%
Config.enable_backprop = True
x = Variable(np.ones((100,100,100)))
y = square(square(square(x)))
y.backward()

Config.enable_backprop = False
x = Variable(np.ones((100,100,100)))
y = square(square(square(x)))
# %%
import contextlib

@contextlib.contextmanager
def using_config(name,value):
    old_value = getattr(Config,name)
    setattr(Config,name,value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

with using_config('enable_backprop',False):
    x = Variable(np.array(2.0))
    y = square(x) 

#%%
def no_grad():
    return using_config('enable_backprop',False)

with no_grad():
    x = Variable(np.array(2.0))
    y = square(x)


# %%
