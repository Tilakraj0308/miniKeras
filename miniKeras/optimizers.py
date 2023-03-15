import numpy as np
class SGD:
    def __init__(self, learning_rate=0.01, momentum=0.0, nesterov=False):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nestrov = nesterov
        self.vel = 0.0
    
    def __call__(self, param):
        if self.nestrov:
            for p in param:
                v_prev = self.vel
                self.vel = self.momentum - self.learning_rate*p.grad
                p.val += -self.momentum*v_prev + (1-self.momentum)*self.vel

        else:
            for p in param:
                self.vel = self.momentum*self.vel - self.learning_rate*p.grad
                p.val += self.vel


class Adagrad:
    def __init__(self, learning_rate=0.1, epsilon=1e-07, cache=0.1):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.cache = cache

    def __call__(self, param):
        for p in param:
            self.cache += p.grad**2
            p.val += (-self.learning_rate*p.grad)/(np.sqrt(self.cache)+self.epsilon)


class RMSprop:
    def __init__(self, learning_rate=0.001, epsilon=1e-07, cache=0.1, decay_rate=0.9):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.cache = cache
        self.decay_rate = decay_rate

    def __call__(self, param):
        for p in param:
            self.cache = self.decay_rate*self.cache + (1-self.decay_rate)*(p.grad**2)
            p.val += (-self.learning_rate*p.grad)/(np.sqrt(self.cache)+self.epsilon)

