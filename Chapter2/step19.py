#%%
import weakref
import numpy as np

class Variable:
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{}은 지원하지 않음".format(type(data)))
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0
        self.name = name

    @property
    def shape(self):
        return self.data.shape
    @property
    def ndim(self):
        return self.data.ndim
    @property
    def size(self):
        return self.data.size
    @property
    def dtypes(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self): ## variable 출력 
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n','\n'+' '*9)
        return 'variable('+p+')'

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

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


#%%
x= Variable(np.array([[1,2,3],[4,5,6]]))
print(x.shape)
# %%
print(len(x))
# %%
print(x)

