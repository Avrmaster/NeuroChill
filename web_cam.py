import cv2
import neural_network as nn
from threading import Thread
import numpy

cam = cv2.VideoCapture(0)
image_width = 10
image_height = 10
train_sets_count = 2


class ImageTrainer(Thread):
    def __init__(self, network, train_res_num):
        super(ImageTrainer, self).__init__()
        self.running = True
        self.network = network
        self.train_res_num = train_res_num
        self.count = 0

    def run(self):
        while self.running:
            self.count += 1

            ret, frame = cam.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frame_array = ((numpy.reshape(cv2.resize(frame, (image_width, image_height)),
                                          image_width * image_height)) / 255) * 0.98 + 0.1

            if self.train_res_num != train_sets_count:  # it's not trained yet
                train_data = numpy.zeros(train_sets_count) + 0.1
                train_data[self.train_res_num] = 0.99
                print("training %s № %d" % (train_data, self.count))

                self.network.train(frame_array.tolist(), train_data.tolist())
            else:
                results = numpy.vectorize(round)(self.network.query(frame_array.tolist()), 4)
                print("results: %s" % results)

                cv2.putText(frame, str(results), (20, 400), 0, 2, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(("train %d" % self.train_res_num) if self.train_res_num < train_sets_count else "querying",
                       frame)
            cv2.waitKey(5)

    def stop(self):
        print("stopping")
        self.running = False


nn = nn.NeuralNetwork(input_count=image_width * image_height, hidden_count=image_width * image_height / 2,
                      output_count=2, bias_needed=True, learning_rate=0.3)

for cutTrain in range(train_sets_count + 1):
    img_taker = ImageTrainer(nn, cutTrain)
    input("Press to start training examples №%d " % cutTrain)
    img_taker.start()
    input("Press to stop ")
    img_taker.stop()
