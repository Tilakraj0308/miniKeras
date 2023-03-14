# from miniKeras.network import Neuron
from miniKeras.engine.miniKeras_units.neuron import Neuron

class Dense:
    def __init__(self, units, activation='tanh', name=''):
        self.units = units
        self.name = name
        self.activation = activation

    def create_layer(self, inp_size):
        self.neurons = [Neuron(inp_size, name=self.name) for i in range(self.units)]
        
    def __call__(self, x):
        outs = [n(x, self.activation) for n in self.neurons]
        return outs[0] if len(outs)==1 else outs

    def parameters(self):
            return [p for n in self.neurons for p in n.parameters()]

    # More classes to add for different type of layers
