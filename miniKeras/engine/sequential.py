from miniKeras.engine.miniKeras_units.unit import Variable
from miniKeras.optimizers import SGD
from miniKeras.optimizers import RMSprop
from miniKeras.optimizers import Adagrad

class Sequential:
    
    def __init__(self, layers):
        self.layers = layers
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        
    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]
    
    def Compile(self, loss='BinaryCrossEntropy', optimizer='SGD'):
        self.loss = loss
        self.optimizer = optimizer
        
        
    def fit(self, x, y, running_status=False):
        inp_size = len(x) if type(x) is list else 1
        for l in self.layers:
            l.create_layer(inp_size)
            inp_size = l.units

        loss = Variable(1000)
        iterations = 1000
        i = 0
        if type(self.optimizer) is str:
            self.optimizer = eval(self.optimizer+'()')
        while loss.val > 0.001 and i < iterations:
            i += 1
            # forward pass:
            y_pred = [self.__call__(x_) for x_ in x]
            loss = sum((y_out-y_gt)**2 for y_out, y_gt in zip(y_pred, y))
            
            # update:
            # for p in self.parameters():
            #     p.val -= self.alpha*p.grad
            self.optimizer.__call__(self.parameters())
            # setting the previous gradients of parameters to zero
            for p in self.parameters():
                p.grad = 0.0

            # backward pass:
            loss.backprop()
            if running_status:
                print("iteration-->",i,"loss=",loss.val)