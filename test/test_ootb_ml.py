from util.util import alias_features

class TestUtil(object):

    def test_alias_features(self):
        assert alias_features(['a', 'b', 'c'], 'target') == ({'a':'x0', 'b':'x1', 'c': 'x2'}, {'target': 'y'})

class TestClassification(object):

    def test_classification(self):
        assert 1 == 1

class TestRegression(object):

    def test_regression(self):
        assert 1 == 1