from kivy.clock import Clock
from kivy.graphics import Color, Rectangle, Line, Translate, Rotate, PushMatrix, PopMatrix
from kivy.uix.widget import Widget
from math import sin, cos, pi, radians
from random import random, choice, randint, randrange
import numpy

def lerp(min_val, max_val, percent):
    return min_val + (max_val - min_val) * percent


def dot_in_polar(angle, length):
    return [length * f(radians(angle)) for f in (cos, sin)]


class NeuralBackground(Widget):
    NEURONS_COUNT = 10

    def redraw(self, dt):
        self.canvas.clear()
        with self.canvas:
            Color(0xC9 / 255, 0xFF / 255, 0xE5 / 255)
            Rectangle(pos=self.pos, size=self.size)  # color background

            for neuron in self.neurons:
                neuron.adjust_size(self.size, delta_time=dt)
                neuron.render(delta_time=dt)
                neuron.draw()

            pass

    def __init__(self, **kwargs):
        Widget.__init__(self, **kwargs)
        self.neurons = []
        for i in range(NeuralBackground.NEURONS_COUNT):
            # keeps spawn position away from window corners
            def constraint_random():
                return 0.15 + 0.7 * random()

            self.neurons.append(DrawnNeuron(rel_position=(constraint_random(), constraint_random()),
                                            rel_size=random() / 20,
                                            all_neurons=self.neurons))

        Clock.schedule_interval(lambda dt: self.redraw(dt), 1 / 60.)
        pass

    pass


class DrawnNeuron:
    DOTS_IN_CIRCLE = 50
    MIN_RECONNECT_TIME = 5
    MAX_RECONNECT_TIME = 20
    MAX_CONNECTIONS = 4

    class ConnectionState:
        def __init__(self, neuron_1, neuron_2):
            self.connected_neurons = (neuron_1, neuron_2)
            self.is_connected = False
            self.connection_progress = 0.0
            self.connect_speed = 4

        def update(self, delta_time):
            self.connection_progress = lerp(self.connection_progress, 1 if self.is_connected else 0,
                                            self.connect_speed * delta_time)

    def __init__(self, rel_position, rel_size, all_neurons):
        if len(rel_position) != 2:
            raise ValueError("%s is invalid position!" % rel_position)
        if type(rel_size) not in [float, int]:
            raise TypeError("%s is invalid size!" % rel_position)

        self._relative_pos = rel_position
        self.pos = numpy.zeros((2, 1))

        self._relative_size = rel_size
        self.size = 0

        self.rotation = 0
        self.rotation_speed = 0.5 * (1 + random())
        self.velocity = numpy.random.randint(0, 5, (2, 1))

        self.connection_states = {}
        self._reconnect_timer = 0
        self._neural_points = []
        for an in range(int(DrawnNeuron.DOTS_IN_CIRCLE)):
            self._neural_points.append(dot_in_polar(360 * an / (DrawnNeuron.DOTS_IN_CIRCLE - 1),
                                                    DrawnNeuron.DOTS_IN_CIRCLE*(0.8+0.2*random())))

        for each_neuron in all_neurons:
            # i'm a new member - and i'll create connection with already existing neuron
            new_connection = DrawnNeuron.ConnectionState(self, each_neuron)
            self.connection_states[each_neuron] = new_connection  # store to myself
            each_neuron.connection_states[self] = new_connection  # store to him
        pass

    def render(self, delta_time):
        self.rotation += delta_time * self.rotation_speed
        # self.pos += self.velocity
        
        for another_neuron, connection_state in self.connection_states.items():
            connection_state.update(delta_time)

        if self._reconnect_timer <= 0:
            for another_neuron, connection_state in self.connection_states.items():
                connection_state.is_connected = False

            for i in range(randrange(DrawnNeuron.MAX_CONNECTIONS)):
                # it's ok to connect with some another neuron twice. Who cares, amount is random anyway
                choice(list(self.connection_states.values())).is_connected = True

            self._reconnect_timer = randint(DrawnNeuron.MIN_RECONNECT_TIME, DrawnNeuron.MAX_RECONNECT_TIME)
            pass
        else:
            self._reconnect_timer -= delta_time
            pass

    def adjust_size(self, window_size, delta_time):
        for n in range(2):
            self.pos[n] = lerp(self.pos[n], window_size[n] * self._relative_pos[n], delta_time)
        self.size = max(window_size) * self._relative_size

    def draw(self):

        PushMatrix()
        Translate(self.pos[0], self.pos[1])
        Rotate(angle=self.rotation * pi)
        Color(0, 0, 0)

        Line(points=self._neural_points, width=2, close=True)
        PopMatrix()

        for another_neuron, connection_state in self.connection_states.items():
            if connection_state.connection_progress != 0:
                Line(points=(list(self.pos), another_neuron.pos), width=4 * connection_state.connection_progress)

        pass
