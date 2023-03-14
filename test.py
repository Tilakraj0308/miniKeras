from miniKeras import network
from miniKeras import optimizers
l = network.Sequential(
    [
        network.layers.Dense(3, name='layer1'),
        network.layers.Dense(2, name='layer2'),
        network.layers.Dense(1, name='layer3'),
    ]
)
opt = optimizers.SGD(learning_rate=0.1)
l.Compile(optimizer=opt)
x = [1,2]
y=[1]
l.fit(x,y, running_status=True)