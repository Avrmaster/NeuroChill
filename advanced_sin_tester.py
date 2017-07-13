from math import *
from random import shuffle

from matplotlib import pyplot

from core import advanced_network

nn = advanced_network.NeuralNetwork((1, 200, 1), 0.3)

samples_cnt = 200000
epoch_cnt = 4


def gen_inputs(sample_number):
    return [(sample_number+1)/(samples_cnt+1)]  # from 0 to 1


def gen_targets(normalized_input):
    x = normalized_input[0]
    result = cos(3*pi*x)  # from -1 to 1
    return [((result+1)/2)*0.9+0.05]

train_data = []
for n in range(samples_cnt):
    train_input = gen_inputs(n)
    train_target = gen_targets(train_input)
    train_pair = [train_input, train_target]
    # print(train_pair)
    train_data.append(train_pair)
    pass
pass


nn.print_weights()
last_prg = -5
for epoch in range(epoch_cnt):
    shuffle(train_data)
    for n in range(samples_cnt):
        cur_prg = round(100*(epoch/epoch_cnt+n/(epoch_cnt*samples_cnt)))
        if cur_prg > last_prg:
            print("\r%d%% training done%s" % (cur_prg, "."*3), flush=False, end="")
            last_prg = cur_prg

        nn.query(train_data[n][0], train_data[n][1])
    pass
nn.print_weights()
print()

outputs = []
last_prg = -5
for n in range(samples_cnt):
    cur_prg = round(100*n/samples_cnt)
    if cur_prg-5 >= last_prg:
        print("\r%d%% drawing done%s" % (cur_prg, "." * 3), flush=True, end="")
        last_prg = cur_prg

    nn_output = nn.query(gen_inputs(n))
    # nn_output = gen_targets(gen_inputs(n))

    outputs.append(nn_output[0]*2-1)
    pass


pyplot.plot(outputs)
pyplot.axis([0, samples_cnt, -1, 1])
pyplot.show()
