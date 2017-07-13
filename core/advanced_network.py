import numpy


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


class NeuralNetwork:
    def __init__(self, configs, learning_rate=0.3):
        """ configs must be an iterable object contains
        3 positive integers - input, hidden and output neurons count"""
        if len(configs) != 3:
            raise ValueError("Configs must contain exactly 3 values (%s passed)!" % len(configs))
        for eachParam in configs:
            if not isinstance(eachParam, int):
                raise TypeError("only integers must be specified in configs!")
            if eachParam <= 0:
                raise ValueError("only positive count of neurons are allowed!")
            pass

        self.in_cnt = configs[0]
        self.hid_cnt = configs[1]
        self.out_cnt = configs[2]
        self.learning_rate = learning_rate

        self.wij = numpy.random.normal(0.0, 2 / (self.in_cnt + self.hid_cnt), (self.hid_cnt, self.in_cnt))
        self.wjk = numpy.random.normal(0.0, 2 / (self.hid_cnt + self.out_cnt), (self.out_cnt, self.hid_cnt))
        self.bj = numpy.random.normal(0.0, 1 / pow(self.hid_cnt, 0.5), (self.hid_cnt, 1))
        self.bk = numpy.random.normal(0.0, 1 / pow(self.out_cnt, 0.5), (self.out_cnt, 1))

        self.activation_function = numpy.vectorize(sigmoid)

        print("Neural network %s with learning rate %f created" % (self.gen_config(), self.learning_rate))
        pass

    def gen_config(self):
        return "(%d:%d:%d)" % (self.in_cnt, self.hid_cnt, self.out_cnt)

    def print_weights(self):
        to_print = "\nWIJ:\n%s\nBJ:\n%s\nWJK:\n%s\nBK:\n%s\n" % (self.wij, self.bj, self.wjk, self.bk)
        print(to_print)
        return to_print

    def query(self, inputs, targets=None):
        if len(inputs) != self.in_cnt:
            raise ValueError(
                "inputs count (%d) do not correspond network's config: %s!" % (len(inputs), self.gen_config()))
        is_training = False
        if targets:
            is_training = True
            if len(targets) != self.out_cnt:
                raise ValueError(
                    "targets count (%d) do not correspond network's config: %s" % (len(targets), self.gen_config()))
            pass

        neural_inputs = numpy.array(inputs, ndmin=2).T

        weighted_hidden_sum = numpy.dot(self.wij, neural_inputs)
        hidden_inputs = weighted_hidden_sum + self.bj
        hidden_outputs = self.activation_function(hidden_inputs)

        weighted_output_sum = numpy.dot(self.wjk, hidden_outputs)
        output_inputs = weighted_output_sum + self.bk  # input of the output layer before activation
        neural_outputs = self.activation_function(output_inputs)

        # print("neural_inputs:\n%s\n" % neural_inputs)
        # print("weighted_hidden_sum:\n%s\n" % weighted_hidden_sum)
        # print("hidden_inputs:\n%s\n" % hidden_inputs)
        # print("hidden_outputs:\n%s\n" % hidden_outputs)
        # print("weighted_output_sum:\n%s\n" % weighted_output_sum)
        # print("output_inputs:\n%s\n" % output_inputs)
        # print("neural_outputs:\n%s\n" % neural_outputs)

        if is_training:
            neural_targets = numpy.array(targets, ndmin=2).T
            output_errors = (neural_targets - neural_outputs)

            #  Error is a sum of 0.5*(Tk-Ok)^2. dE/Ok = -sum(Tk-Ok)
            #  Ok = sigmoid(weighted sum + BiasK)
            #  derivative of sigmoid(x) is sigmoid(x)*(1-sigmoid(x)1)
            output_derivatives = output_errors * neural_outputs * (1.0 - neural_outputs)
            #  applying chain rule: d(weighted sum + BiasK)/dBk = 1
            bk_deltas = output_derivatives * 1
            #  applying chain rule: d(weighted sum + BiasK)/dWjk = Oj
            wjk_deltas = numpy.dot(bk_deltas, hidden_outputs.T)

            #  d(Weighted sum + BiasK)/dWij equals to sum of output derivatives
            #  multiplied by corresponding
            #  applying chain rule: d(weighted sum + BiasK)/dBj
            bj_deltas = numpy.dot(self.wjk.T, output_derivatives)*hidden_outputs*(1.0 - hidden_outputs)
            wij_deltas = numpy.dot(bj_deltas, neural_inputs.T)

            # print("\n\n%s\n%s\n%s\n%s\n\n\n" % (wij_deltas, bj_deltas, wjk_deltas, bk_deltas))

            self.wjk += self.learning_rate * wjk_deltas
            self.bk += self.learning_rate * bk_deltas
            self.wij += self.learning_rate * wij_deltas
            self.bj += self.learning_rate * bj_deltas
            pass

        return numpy.transpose(neural_outputs)
