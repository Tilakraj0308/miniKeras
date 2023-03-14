class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def __call__(self, param):
        for p in param:
            p.val -= self.learning_rate*p.grad

