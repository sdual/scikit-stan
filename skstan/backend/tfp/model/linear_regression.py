import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import edward2 as ed

from skstan.backend.tfp.model.base_model import BaseTFPModel


class TFPLinearRegression(BaseTFPModel):
    """
    Linear regression implementation using TensorFlow Probability.
    """

    def __init__(self, scale: float):
        self._scale: float = scale

    def posterior_dist(self, features):
        """

        Parameters
        ----------
        features

        Returns
        -------

        """
        regression = tfp.sts.LinearRegression(
            design_matrix=tf.stack([]),
            weights_prior=tfd.Normal(loc=0.0, scale=self._scale)
        )


class TFPLogisticRegression(BaseTFPModel):
    """
    Logistic regression implementation using TensorFlow Probability.
    """

    def __init__(self):
        pass

    def set_params(self):
        pass

    def posterior_dist(self, features) -> tfd.Distribution:
        """

        Parameters
        ----------
        features

        Returns
        -------

        """
        # TODO: implement Logisitc regression.
        pass


class TFPPoissonRegression(BaseTFPModel):
    """
    Poisson regression implementation using TensorFlow Probability.
    """

    def __init__(self):
        pass

    def posterior_dist(self, features):
        """

        Parameters
        ----------
        features

        Returns
        -------

        """
        # TODO: implement Poission regression.
        pass
