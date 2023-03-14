import random
# from miniKeras.unit import Variable
# from miniKeras_units.unit import Variable
from miniKeras.engine.miniKeras_units.unit import Variable
class Neuron:
    
    def __init__(self, inp_size, name=''):
        self.w = [Variable(random.uniform(-1,1), name=name) for i in range(inp_size)]
        self.b = Variable(random.uniform(-1,1))
    

    def __call__(self, x, activation='tanh'):
        if type(x) is not list:
            lis = []
            lis.append(x)
            x = lis.copy()
        res = sum((i*j for i,j in zip(self.w, x)), self.b)
        assert activation in ['relu', 'tanh']
        fin = 'res.'+activation+'()'
        return eval(fin)

    def parameters(self):
        return self.w+[self.b]
