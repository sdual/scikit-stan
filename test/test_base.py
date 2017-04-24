from unittest import TestCase

import numpy as np

from skstan.base import BaseStanData


class TestBaseStanData(TestCase):

    def test_append(self):
        init_dict = {
            'x': np.ndarray([[1, 2, 3], [4, 5, 6]]),
            'shrinkage': 10
        }

        test_dict = {
            'x': np.ndarray([[7, 8, 9], [10, 11, 12]]),
            'y': np.ndarray([1, 2, 3]),
            'shrinkage': 12
        }

        base_data = BaseStanData(init_dict)

        result = base_data.append(test_dict)
        self.assertEqual
