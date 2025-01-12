class StateMachine:
    def __init__(self):
        self.state = None

    def run(self):
        self.state.run()

class State:
    def __init__(self):
        pass