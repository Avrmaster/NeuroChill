import kivy.graphics as graph
from kivy.app import App
from kivy.core.window import Window
from gui.background_widgets import NeuralBackground


class MyApp(App):
    def __init__(self, **kwargs):
        Window.size = (720, 480)
        App.__init__(self, **kwargs)
        pass

    def build(self):
        return NeuralBackground()

    def on_pause(self):
        print("on pause")
        pass

    def on_resume(self):
        print("on resume")
        pass

    def on_stop(self):
        print("on stop")
        pass


# help(graph)
MyApp().run()
