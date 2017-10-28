from unittest import TestCase

from skstan.stan.omm import Int
from skstan.stan.omm import Matrix
from skstan.stan.omm import StanCode


class TestStanCode(TestCase):

    def test_parenthesis(self):
        """
        Check that parenthesis is added.
        """
        class SampleStanCode(StanCode):
            REP = 'data'

        test_code = SampleStanCode()
        code = test_code._parenthesis('test code')

        expected = 'data{test code}'
        self.assertEqual(code, expected)

    def test_default_render(self):
        """
        Check stan code is rendered.
        """
        class SampleStanCode(StanCode):
            REP = 'data'

            def __init__(self, *variables):
                super().__init__(*variables)

        int_code1 = Int('m')
        int_code2 = Int('n')
        matrix_code = Matrix('x', int_code1, int_code2)

        test_code = SampleStanCode(matrix_code)
        rendered = test_code._default_render()

        expected = 'data{int m;int n;matrix[m, n] x;}'
        self.assertEqual(rendered, expected)