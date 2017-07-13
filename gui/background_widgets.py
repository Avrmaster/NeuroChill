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


constraint = numpy.vectorize(
    lambda val, min_val, max_val: min_val if val < min_val else max_val if val > max_val else val)


class NeuralBackground(Widget):
    NEURONS_COUNT = 3

    def redraw(self, dt):
        self.canvas.clear()
        with self.canvas:
            Color(0xC9 / 255, 0xFF / 255, 0xE5 / 255)
            Rectangle(pos=self.pos, size=self.size)  # color background

            for neuron in self.neurons:
                neuron.window_size = self.size
                neuron.render(delta_time=dt)
                neuron.draw()

            pass

    def __init__(self, **kwargs):
        Widget.__init__(self, **kwargs)
        self.neurons = []
        for i in range(NeuralBackground.NEURONS_COUNT):
            self.neurons.append(DrawnNeuron(all_neurons=self.neurons))

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

        self.pos = numpy.zeros((2, 1), dtype="float64")
        self._window_size = numpy.zeros((2, 1))

        self.size = 0
        self.rel_size = 0.05 + 0.35 * random()  # size relatively to window

        self.rotation = 0
        self.rotation_speed = 0.5 * (1 + random())
        self.velocity = numpy.zeros((2, 1), dtype="float64")

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
        pass

    def intersect(self, another_neuron):
        return sum((self.pos-another_neuron.pos)**2) < self.size+another_neuron.size

    def run_away_vector(self, another_neuron):
        return (self.size+another_neuron.size) - (another_neuron.pos - self.pos)

    def render(self, delta_time):
        self.size = DrawnNeuron.DOTS_IN_CIRCLE

        self.rotation += delta_time * self.rotation_speed
        self.velocity = numpy.add(self.velocity, 0.5*(self.window_size / 2 - self.pos))

        for each_neuron in self.connection_states.keys():
            if self.intersect(each_neuron):
                self.velocity = numpy.add(self.velocity, 0.8*self.run_away_vector(each_neuron))
                pass

        self.velocity = constraint(self.velocity, -DrawnNeuron.MAX_SPEED, DrawnNeuron.MAX_SPEED)
        self.pos += self.velocity*delta_time

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

    def get_window_size(self):
        return self._window_size

    def set_window_size(self, size):
        for n in range(2):
            self._window_size[n] = size[n]

    window_size = property(get_window_size, set_window_size)
