import tensorflow_probability as tfp

from skstan.backend.tfp.model import TFPLinearRegression
from skstan.backend.tfp.model import TFPLogisticRegression
from skstan.backend.tfp.model import TFPPoissonRegression


class TFPBackend:
    """
    TensorFlow backend class.
    This class provides model clases which are implemented using TensorFlow
    Probability.
    """

    LinearRegression = TFPLinearRegression
    LogisitcRegression = TFPLogisticRegression
    PoissionRegression = TFPPoissonRegression


_tfd = tfp.distributions


class TFDistribution:
    """
    TensorFlow Distribution class.

    This class provides typical distributions.
    """

    @staticmethod
    def normal(mean, std) -> _tfd.Normal:
        """
        Normal distribution object.

        Parameters
        ----------
        mean: float or int
            The mean value of Normal distribution.
        std: float or int
            The standard deviation of Normal distribution.
        Returns
            The Normal distribution instance of TensorFlow Probability.
        -------

        """
        pass

    @staticmethod
    def bernoulli(probs=()) -> _tfd.Bernoulli:
        """

        Parameters
        ----------
        probs

        Returns
        -------

        """
        pass

    def binomial(self) -> _tfd.Binomial:
        pass
