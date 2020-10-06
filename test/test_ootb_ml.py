from util.util import alias_features
import pandas as pd

class TestUtil(object):

    def test_alias_features(self):
        df = pd.DataFrame([[1,2,3,4,5], [4,5,6,7,8]], columns=['a','b','c','d','z'])
        df, alias = alias_features(df, features=['a','b','c'], target='z')
        assert list(df.columns) == ['x0','x1','x2','d','y']

class TestClassification(object):

    def test_classification(self):
        assert 1 == 1

class TestRegression(object):

    def test_regression(self):
        assert 1 == 1