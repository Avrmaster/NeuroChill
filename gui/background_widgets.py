from kivy.clock import Clock
from kivy.graphics import Color, Rectangle, Line, Translate, Ellipse, PushMatrix, PopMatrix
from kivy.uix.widget import Widget
from math import sin, cos, pi, radians
from random import random, choice, randint, randrange
import numpy
import pymunk


def lerp(min_val, max_val, percent):
    return min_val + (max_val - min_val) * percent


def dot_in_polar(angle, length):
    return [length * f(radians(angle)) for f in (cos, sin)]


constraint = numpy.vectorize(
    lambda val, min_val, max_val: min_val if val < min_val else max_val if val > max_val else val)


class NeuralBackground(Widget):
    NEURONS_COUNT = 100

    def redraw(self, dt):

        static_lines = [
            pymunk.Segment(self.space.static_body, a, b, radius=1)
            for a, b in [
                [(0, 0), (self.size[0], 0)],  # bottom
                [(0, 0), (0, self.size[1])],  # left
                [(0, self.size[1]), (self.size[0], self.size[1])],  # top
                [(self.size[0], 0), (self.size[0], self.size[1])],  # right
            ]
        ]

        self.space.add(static_lines)

        self.space.step(dt)
        self.canvas.clear()
        with self.canvas:
            Color(0xC9 / 255, 0xFF / 255, 0xE5 / 255)
            Rectangle(pos=self.pos, size=self.size)  # color background

            for neuron in self.neurons:
                neuron.window_size = self.size
                neuron.render(delta_time=dt)
                neuron.draw()

            pass
        self.space.remove(static_lines)

    def __init__(self, **kwargs):
        Widget.__init__(self, **kwargs)

        self.space = pymunk.Space()
        self.space.damping = 0.99

        self.neurons = []
        for i in range(NeuralBackground.NEURONS_COUNT):
            new_neuron = DrawnNeuron(all_neurons=self.neurons)
            self.space.add(new_neuron.body, new_neuron.shape)
            self.neurons.append(new_neuron)

        Clock.schedule_interval(lambda dt: self.redraw(dt), 1 / 60.)
        pass

    pass


class DrawnNeuron:
    DOTS_IN_CIRCLE = 50
    MIN_RECONNECT_TIME = 5
    MAX_RECONNECT_TIME = 20
    MAX_CONNECTIONS = 4
    MAX_SPEED = 80

    class ConnectionState:
        def __init__(self, neuron_1, neuron_2):
            self.connected_neurons = (neuron_1, neuron_2)
            self.is_connected = False
            self.connection_progress = 0.0
            self.connect_speed = 4

        def update(self, delta_time):
            self.connection_progress = lerp(self.connection_progress, 1 if self.is_connected else 0,
                                            self.connect_speed * delta_time)

    def __init__(self, all_neurons):

        self._window_size = numpy.zeros((2, 1))
        self.rel_size = 0.05 + 0.35 * random()  # size relatively to window

        self.rotation_speed = 0.5 * (1 + random())

        self.connection_states = {}
        self._reconnect_timer = 0
        self._neural_points = []
        for an in range(int(DrawnNeuron.DOTS_IN_CIRCLE)):
            self._neural_points.append(dot_in_polar(360 * an / (DrawnNeuron.DOTS_IN_CIRCLE - 1),
                                                    DrawnNeuron.DOTS_IN_CIRCLE * (0.8 + 0.2 * random())))

        for each_neuron in all_neurons:
            # i'm a new member - and i'll create connection with already existing neuron
            new_connection = DrawnNeuron.ConnectionState(self, each_neuron)
            self.connection_states[each_neuron] = new_connection  # store to myself
            each_neuron.connection_states[self] = new_connection  # store to him

        self.body = pymunk.Body(1, 1500)
        self.body.position = 50, 50  # to force it go into the center
        self.shape = pymunk.Circle(body=self.body, radius=50)
        self.shape.elasticity = 0.9999999
        pass

    def render(self, delta_time):

        for another_neuron, connection_state in self.connection_states.items():
            connection_state.update(delta_time)

        if self._reconnect_timer <= 0:
            for another_neuron, connection_state in self.connection_states.items():
                connection_state.is_connected = False

            if len(self.connection_states) > 1:
                for i in range(randrange(DrawnNeuron.MAX_CONNECTIONS)):
                    # it's ok to connect with some another neuron twice. Who cares, amount is random anyway
                    choice(list(self.connection_states.values())).is_connected = True

            self._reconnect_timer = randint(DrawnNeuron.MIN_RECONNECT_TIME, DrawnNeuron.MAX_RECONNECT_TIME)
            pass
        else:
            self._reconnect_timer -= delta_time
            pass

    def draw(self):

        PushMatrix()
        Translate(self.body.position[0], self.body.position[1])
        # Rotate(angle=self.rotation * pi)
        Color(0, 0, 0)

        Line(points=self._neural_points, width=2, close=True)
        PopMatrix()
        return

        for another_neuron, connection_state in self.connection_states.items():
            if connection_state.connection_progress != 0:
                Line(points=(list(self.pos), another_neuron.pos), width=4 * connection_state.connection_progress)

        pass

    def get_window_size(self):
        return self._window_size

    def set_window_size(self, size):
        for n in range(2):
            self._window_size[n] = size[n]

    window_size = property(get_window_size, set_window_size)
