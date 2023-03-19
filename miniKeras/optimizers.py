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
    def __init__(self, learning_rate=0.001, epsilon=1e-07, cache=0.1):
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

class Adam:
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.99, epsilon=1e-07):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.momentum = 0.0
        self.vel = 0.0
        self.t = 1

    def __call__(self, param):
        for p in param:

            self.momentum = self.beta_1*self.momentum+(1-self.beta_1)*p.grad
            self.vel = self.beta_2*self.vel+(1-self.beta_2)*(p.grad**2)
            #update momentum and vel
            self.momentum /= 1-self.beta_1**self.t
            self.vel /= 1-self.beta_2**self.t
            p.val += (-self.learning_rate*self.momentum)/(np.sqrt(self.vel)+self.epsilon)
            self.t += 1

class AdamW:
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.99, epsilon=1e-07, decay_rate=0.004):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.momentum = 0.0
        self.vel = 0.0
        self.t = 1
        self.decay_rate = decay_rate

    def __call__(self, param):
        #decaying learning_rate over time
        self.learning_rate *= (1-self.decay_rate)
        for p in param:
            self.momentum = self.beta_1*self.momentum+(1-self.beta_1)*p.grad
            self.vel = self.beta_2*self.vel+(1-self.beta_2)*(p.grad**2)
            #update momentum and vel
            self.momentum /= 1-self.beta_1**self.t
            self.vel /= 1-self.beta_2**self.t
            p.val += (-self.learning_rate*self.momentum)/(np.sqrt(self.vel)+self.epsilon)
            # print(self.learning_rate)
            self.t += 1
