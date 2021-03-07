from auto_ml.util import train_val_test_split, pretty_confusion_matrix
import pandas as pd

class TestUtil(object):

    def test_pretty_confusion_matrix(self):
        y_true = [1,1,0,0,0,0,1]
        y_pred = [1,1,1,0,0,0,0]
        cm = pretty_confusion_matrix(y_true, y_pred)
        assert isinstance(cm, pd.DataFrame)
        assert cm.loc['Actual 0', 'Predicted 0'] == 3
        assert cm.loc['Actual 0', 'Predicted 1'] == 1
        assert cm.loc['Actual 1', 'Predicted 0'] == 1
        assert cm.loc['Actual 1', 'Predicted 1'] == 2

    def test_train_val_test_split(self):
        train, val, test = ( .75, .15, .10 )
        df = pd.DataFrame({'x0': range(0,100), 'x1': range(0,100), 'y': range(0,100)})
        X = df[['x0','x1']]
        y = df['y']
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, train, val, test)
        assert X_train.shape[0] == train * df.shape[0]
        assert X_val.shape[0] == val * df.shape[0]
        assert X_test.shape[0] == test * df.shape[0]
        assert not set(X_train.index) & set(X_val.index) & set(X_test.index)
        assert not set(y_train.index) & set(y_val.index) & set(y_test.index)

        

