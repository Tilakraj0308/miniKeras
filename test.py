# from miniKeras import network
# from miniKeras import optimizers
# from miniKeras import layers

import miniKeras as mk
l = mk.Sequential(
    [
        mk.layers.Dense(3, name='layer1'),
        mk.layers.Dense(2, name='layer2'),
        mk.layers.Dense(1, name='layer3'),
    ]
)
# opt = mk.optimizers.SGD()
# opt = mk.optimizers.Adagrad()
# opt = mk.optimizers.RMSprop()
opt = mk.optimizers.Adam()


l.Compile(optimizer=opt)
x = [1,2]
y=[1]
l.fit(x,y, running_status=True)