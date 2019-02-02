from skstan.distribution import BaseDistribution


class Normal(BaseDistribution):

    def __init__(self, mean, std):
        self._mean = mean
        self._std = std

    def sample(self):
        pass
