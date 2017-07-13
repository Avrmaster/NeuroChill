from core.advanced_network import *

nn = NeuralNetwork([2, 4, 2], learning_rate=0.3)

trainData = ((0.1, 0.1, 0.9, 0.1), (0.1, 1, 0.1, 0.9), (1, 0.1, 0.1, 0.9), (1, 1, 0.1, 0.1))
for n in range(100000):
    test_input = numpy.random.rand(1, 2)
    correct = numpy.logical_xor(round(test_input[0][0]), round(test_input[0][1]))
    _targets = [0.01, 0.99] if correct else [0.99, 0.01]  # first neuron blasts when 0 should be an answer

    # test_case = n % 4
    # test_input = [trainData[test_case][:-2]]
    # _targets = trainData[test_case][-2:]

    nn.query(test_input[0], _targets)


success_cnt = 0
tries = 2000
for n in range(tries):
    test_input = numpy.random.rand(1, 2)

    # test_case = n % 4
    # test_input = [trainData[test_case][:-2]]

    correct = numpy.logical_xor(round(test_input[0][0]), round(test_input[0][1]))

    prediction = nn.query(test_input[0])
    test_condition = "Test input: %s, prediction: %s" % (test_input, prediction)

    prediction = 0 if (prediction[0][0] > prediction[0][1]) else 1

    if bool(prediction) == bool(correct):
        success_cnt += 1
        print(test_condition)
    else:
        print("%s -- WRONG" % test_condition)
        pass


print("\nSuccess rate: %d" % (100 * success_cnt / tries))
