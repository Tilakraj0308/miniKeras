from miniKeras import network

l = network.Sequential(
    [
        network.layers.Dense(3, name='layer1'),
        network.layers.Dense(2, name='layer2'),
        network.layers.Dense(1, name='layer3'),
    ]
)
l.Compile()
x = [1,2]
y=[1]
l.fit(x,y, running_status=True)