from util.util import alias_features, check_features_exist
import pandas as pd
import pytest

class TestUtil(object):

    def test_alias_features(self):
        df = pd.DataFrame([[1,2,3,4,5], [4,5,6,7,8]], columns=['a','b','c','d','z'])
        df, alias = alias_features(df, features=['a','b','c'], target='z')
        assert list(df.columns) == ['x0','x1','x2','d','y']

    def test_check_features_exist(self):
        df = pd.DataFrame([[1,2,3,4,5], [4,5,6,7,8]], columns=['a','b','c','d','z'])
        feats = ['a', 'b', 'c']
        assert check_features_exist(df, feats)
        feats1 = ['b','c','d','z','a']
        assert check_features_exist(df, feats1)
        feats2 = ['a','b','C']
        with pytest.raises(ValueError):
            check_features_exist(df, feats2)

class TestClassification(object):

    def test_classification(self):
        assert 1 == 1

class TestRegression(object):

    def test_regression(self):
        assert 1 == 1