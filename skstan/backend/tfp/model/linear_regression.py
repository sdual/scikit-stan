import tensorflow as tf
import tensorflow_probability as tfp

from skstan.backend.tfp.model.base_model import BaseTFPModel

ed = tfp.edward2


class TFPLinearRegression(BaseTFPModel):
    """
    Linear regression implementation using TensorFlow Probability.
    """

    def __init__(self, chains: int, warmup: int, shrinkage: int, n_jobs: int,
                 n_itr: int, algorithm: str = 'NUTS', verbose: bool = False):
        self._shrinkage = shrinkage
        self._chains = chains

    def _validate_params(self):
        pass

    def posterior_dist(self, features):
        """

        Parameters
        ----------
        features

        Returns
        -------

        """
        pass


class TFPLogisticRegression(BaseTFPModel):
    """
    Logistic regression implementation using TensorFlow Probability.
    """

    def __init__(self, fit_intercept=True):
        self._fit_intercept = fit_intercept
        self._validate_params()

    def _validate_params(self):
        pass

    def posterior_dist(self, features):
        """

        Parameters
        ----------
        features

        Returns
        -------

        """
        pass
        # coeffs = ed.Normal(loc=tf.zeros(features.shape[1]),
        #                    scale=1.0,
        #                    name='coeffs')
        #
        # logits = tf.tensordot(features, coeffs, [[1], [0]])
        # if self._fit_intercept:
        #     bias = ed.Normal(loc=0.0, scale=1.0, name='bias')
        #     logits = logits + bias
        # target_dist = ed.Bernoulli(logits=logits)
        # return target_dist

    def _set_prior_to_posterior_mean(self):
        pass
        # name = kwargs.get("name")
        # if name == "w":
        #     return posterior_w.distribution.mean()
        # elif name == "b":
        #     return posterior_b.distribution.mean()
        # return f(*args, **kwargs)


class TFPPoissonRegression(BaseTFPModel):
    """
    Poisson regression implementation using TensorFlow Probability.
    """

    def __init__(self):
        pass

    def _validate_params(self):
        pass

    def posterior_dist(self, features):
        """

        Parameters
        ----------
        features

        Returns
        -------

        """
        pass
