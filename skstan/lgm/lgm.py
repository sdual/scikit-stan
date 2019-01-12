from abc import ABCMeta, abstractmethod


class BaseLinearRegression(metaclass=ABCMeta):
    """
    Abstract base linear regression model class.
    """

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass
