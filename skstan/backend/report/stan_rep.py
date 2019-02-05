from pystan import StanModel


class StanInferenceReport:

    def __init__(self, pystan: StanModel):
        self._pysta = pystan

    def print(self):
        pass

    def print_oroginal(self):
        pass
