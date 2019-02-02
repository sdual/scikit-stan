from abc import ABCMeta, abstractmethod


class BaseDistribution(metaclass=ABCMeta):

    @abstractmethod
    def sample(self):
        pass
