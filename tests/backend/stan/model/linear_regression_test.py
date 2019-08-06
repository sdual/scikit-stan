from skstan.backend.stan.model import StanLinearRegression
from skstan.backend.stan.model import StanLogisticRegression
from skstan.backend.stan.model import StanPoissonRegression
from skstan.backend.stan.model.base_model import StanModelLoadMixin
from skstan.params import StanLinearRegressionParams


class TestStanLinearRegression:
    def test_fit(self, monkeypatch):
        # Test that a stan model is trained by using `fit` method.
        params = StanLinearRegressionParams(
            chains=3,
            warmup=1000,
            n_itr=5000,
            n_jobs=1,
            algorithm='NUTS',
            verbose=False,
            shrinkage=10,
        )

        def mock_laod_model(dummy):
            return None

        monkeypatch.setattr(StanLinearRegresson, 'load_model', mock_load_model)

        slr = StanLinearRegression(params)
        X = np.array([[5.1, 3.5, 1.4, 0.2], [5.4, 3.4, 1.7, 0.2],
                      [7., 3.2, 4.7, 1.4], [5., 2., 3.5, 1.],
                      [5.9, 3.2, 4.8, 1.8], [6.3, 3.3, 6., 2.5],
                      [6.9, 3.2, 5.7, 2.3]])

        y = np.array([
            0,
            0,
            1,
            1,
            1,
            2,
            2,
        ])

        slr.fit(X, y)

    def test_get_model_file_name(self, monkeypatch):
        # Test that model file name is returned.

        def mock_load_model(dummy):
            return None

        monkeypatch.setattr(StanModelLoadMixin, 'load_model', mock_load_model)
        params = StanLinearRegressionParams(
            chains=3,
            warmup=1000,
            n_itr=5000,
            n_jobs=1,
            algorithm='NUTS',
            verbose=False,
            shrinkage=10,
        )
        slr = StanLinearRegression(params)
        actual = slr.get_model_file_name()
        expected = 'linear_regression.pkl'
        assert actual == expected


class TestStanLogisticRegression:
    def test_get_model_file_name(self, monkeypatch):
        # Test that model file name is returned.
        def mock_load_model(dummy):
            return None

        monkeypatch.setattr(StanModelLoadMixin, 'load_model', mock_load_model)
        slr = StanLogisticRegression(chains=3,
                                     warmup=1000,
                                     shrinkage=10,
                                     n_jobs=1,
                                     n_itr=5000)
        actual = slr.get_model_file_name()
        expected = 'logistic_regression.pkl'
        assert actual == expected


class TestStanPoissionRegression:
    def test_get_model_file_name(self, monkeypatch):
        # Test that model file name is returned.
        def mock_load_model(dummy):
            return None

        monkeypatch.setattr(StanModelLoadMixin, 'load_model', mock_load_model)
        spr = StanPoissonRegression(chains=3,
                                    warmup=1000,
                                    shrinkage=10,
                                    n_jobs=1,
                                    n_itr=5000)
        actual = spr.get_model_file_name()
        expected = 'poisson_regression.pkl'
        assert actual == expected
