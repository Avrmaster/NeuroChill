from kivy.clock import Clock
from kivy.graphics import Color, Rectangle, Line, Translate, Rotate, PushMatrix, PopMatrix
from kivy.uix.widget import Widget
from kivy.core.window import Window, WindowBase
from math import sin, cos, pi, radians, atan2
from random import random, choice, randint, randrange
import numpy
import pymunk
from functools import reduce


def lerp(min_val, max_val, percent):
    return min_val + (max_val - min_val) * percent


def dot_in_polar(angle, length):
    return [length * f(radians(angle)) for f in (cos, sin)]


constraint = numpy.vectorize(
    lambda val, min_val, max_val: min_val if val < min_val else max_val if val > max_val else val)

space = pymunk.Space()


class NeuralBackground(Widget):
    NEURONS_COUNT = 10
    BORDER_OFFSET = 20  # distance at which neurons can't come closer to borders or the window

    def __init__(self, **kwargs):
        Widget.__init__(self, **kwargs)

        # space.damping = 0.99

        self.neurons = []
        for i in range(NeuralBackground.NEURONS_COUNT):
            new_neuron = DrawnNeuron(all_neurons=self.neurons)
            space.add(new_neuron.body, new_neuron.shape)
            self.neurons.append(new_neuron)

        Clock.schedule_interval(lambda dt: self.redraw(dt), 1 / 60.)

    def redraw(self, dt):
        for n in range(2):
            self.size[n] = max(self.size[n], 100)

        static_lines = [
            pymunk.Segment(space.static_body, a, b, radius=1)
            for a, b in [
                [(0, 0), (self.size[0], 0)],  # bottom
                [(0, 0), (0, max(100, self.size[1]))],  # left
                [(0, self.size[1]), (self.size[0], self.size[1])],  # top
                [(self.size[0], 0), (self.size[0], self.size[1])],  # right
            ]
        ]

        space.add(static_lines)

        space.step(dt/5)
        self.canvas.clear()
        with self.canvas:
            Color(0xC9 / 255, 0xFF / 255, 0xE5 / 255)
            Rectangle(pos=self.pos, size=self.size)  # color background

            for neuron in self.neurons:
                neuron.window_size = self.size
                neuron.render(delta_time=dt)
                neuron.draw()

            pass
        space.remove(static_lines)


class DrawnNeuron:
    DOTS_IN_CIRCLE = 50
    MIN_RECONNECT_TIME = 5
    MAX_RECONNECT_TIME = 20
    MAX_CONNECTIONS = 4
    MAX_SPEED = 80
    MAX_FORCE = 5

    class ConnectionState:
        def __init__(self, neuron_1, neuron_2):
            joints_count = 3

            self.connected_neurons = (neuron_1, neuron_2)
            self._is_connected = False
            self.connection_progress = 0.0
            self.connect_speed = 4
            self.neurons = (neuron_1, neuron_2)
            self.joints = [pymunk.DampedSpring(neuron_1.body, neuron_2.body,
                                               (n * 50 / joints_count, n * 50 / joints_count),
                                               (n * 50 / joints_count, n * 50 / joints_count),
                                               rest_length=0, stiffness=5, damping=5)
                           for n in range(joints_count)]

        def update(self, delta_time):
            self.connection_progress = lerp(self.connection_progress, 1 if self._is_connected else 0,
                                            self.connect_speed * delta_time)
            for joint in self.joints:
                joint.rest_length = 2 * reduce(lambda x, y: x.size + y.size, self.neurons)

        def is_connected(self):
            return self._is_connected

        def set_connected(self, is_connected):
            self._is_connected = is_connected
            if is_connected:
                for joint in self.joints:
                    if joint not in space.constraints:
                        space.add(joint)
            else:
                for joint in self.joints:
                    if joint in space.constraints:
                        space.remove(joint)
            pass

        is_connected = property(is_connected, set_connected)

    def __init__(self, all_neurons):
        self._window_size = numpy.zeros((2, 1))

        self.size = 0
        self.body = pymunk.Body(1, 1500)
        self.body.position = 50, 50  # to force it go into the center
        self.shape = pymunk.Circle(body=self.body, radius=self.size)
        self.shape.elasticity = 0
        self.shape.friction = 0.2

        self.surface_beam_angles = []
        self.init_beam_angles()

        self.connection_states = {}
        self._reconnect_timer = 0
        self._neural_shape_lengths = []
        self.init_neural_shape()

        self._neural_points = []
        for an in range(int(DrawnNeuron.DOTS_IN_CIRCLE)):
            self._neural_points.append(dot_in_polar(0, 0))

        for each_neuron in all_neurons:
            # i'm a new member - and i'll create connection with already existing neuron
            new_connection = DrawnNeuron.ConnectionState(self, each_neuron)
            self.connection_states[each_neuron] = new_connection  # store to myself
            each_neuron.connection_states[self] = new_connection  # store to him
        pass

    def render(self, delta_time):
        window_center = list(map(lambda x: x / 2, self.window_size))
        neuron_pos = [self.body.position[n] for n in range(2)]
        center_force = constraint([window_center[n] - neuron_pos[n] for n in range(2)],
                                  -DrawnNeuron.MAX_FORCE,
                                  DrawnNeuron.MAX_FORCE)

        self.body.apply_force_at_world_point(center_force, (0, 0))

        self.body.position = [constraint(self.body.position[n], NeuralBackground.BORDER_OFFSET,
                                         max(100, self.window_size[n]) - NeuralBackground.BORDER_OFFSET)
                              for n in range(2)]

        for another_neuron, connection_state in self.connection_states.items():
            connection_state.update(delta_time)

        if self._reconnect_timer <= 0:
            for another_neuron, connection_state in self.connection_states.items():
                connection_state.is_connected = False

            if len(self.connection_states) > 0:
                for i in range(randrange(DrawnNeuron.MAX_CONNECTIONS)):
                    # it's ok to connect with some another neuron twice. Who cares, amount is random anyway
                    choice(list(self.connection_states.values())).is_connected = True
                    pass

            self.init_beam_angles()
            self._reconnect_timer = randint(DrawnNeuron.MIN_RECONNECT_TIME, DrawnNeuron.MAX_RECONNECT_TIME)
            pass
        else:
            self._reconnect_timer -= delta_time
            pass

        self.size = lerp(self.size, self.window_size[0] / (2 * NeuralBackground.NEURONS_COUNT), delta_time)
        self.shape.unsafe_set_radius(self.size)
        self.init_neural_shape()

        for n in range(DrawnNeuron.DOTS_IN_CIRCLE):
            goal_dot = dot_in_polar(360 * n / (DrawnNeuron.DOTS_IN_CIRCLE - 1),
                                    self.size * self._neural_shape_lengths[n])
            for i in range(2):
                self._neural_points[n][i] = lerp(self._neural_points[n][i], goal_dot[i], delta_time)

    def init_beam_angles(self):
        self.surface_beam_angles = []
        for n in range(randint(3, 8)):
            self.surface_beam_angles.append(randrange(DrawnNeuron.DOTS_IN_CIRCLE))

    def init_neural_shape(self):
        self._neural_shape_lengths = []

        for an in range(int(DrawnNeuron.DOTS_IN_CIRCLE)):
            self._neural_shape_lengths.append(0.2)
        for beam in self.surface_beam_angles:
            for n in range(5):
                self._neural_shape_lengths[beam % DrawnNeuron.DOTS_IN_CIRCLE] = 0.6 + 0.2 * sin(n*pi/5)

        angle = -self.body.angle
        if angle < 0:
            angle += 360

        # p = (angle*DrawnNeuron.DOTS_IN_CIRCLE/ 360)
        # self._neural_shape_lengths[int(p)] = 3
        # print("angle: %f, p: %d" % (angle, p))

        for another_neuron, connection_state in self.connection_states.items():
            if connection_state.is_connected:
                connection_point = (atan2(another_neuron.body.position[1] - self.body.position[1],
                                          another_neuron.body.position[0] - self.body.position[0]) + radians(angle)) \
                                   * DrawnNeuron.DOTS_IN_CIRCLE / (2*pi)

                points_to_adjust_count = max(1, int(DrawnNeuron.DOTS_IN_CIRCLE / 20))
                for n in range(points_to_adjust_count):
                    self._neural_shape_lengths[int((-points_to_adjust_count / 2 + n + connection_point)
                                               % DrawnNeuron.DOTS_IN_CIRCLE)] =\
                        1 + 2 * sin(pi * n / points_to_adjust_count)

                pass

    def draw(self):

        PushMatrix()
        Translate(self.body.position[0], self.body.position[1])
        Rotate(angle=self.body.angle)
        Color(0, 0, 0)

        # Line(points=(0, 0, self.size, 0), width=1)
        Line(points=self._neural_points, width=0.5, close=True)
        PopMatrix()

        # for another_neuron, connection_state in self.connection_states.items():
        #     if connection_state.connection_progress != 0:
        #         Line(points=(list(self.body.position), another_neuron.body.position),
        #              width=2 * connection_state.connection_progress)

        pass

    def get_window_size(self):
        return self._window_size

    def set_window_size(self, size):
        self._window_size = [size[n] for n in range(2)]

    window_size = property(get_window_size, set_window_size)
