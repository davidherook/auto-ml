from auto_ml.auto_ml import AutoMLClassifier
import pandas as pd

class TestAutoML(object):

    clf = AutoMLClassifier()

    def test_pretty_confusion_matrix(self):
        y_true = [1,1,0,0,0,0,1]
        y_pred = [1,1,1,0,0,0,0]
        cm = self.clf.pretty_confusion_matrix(y_true, y_pred)
        assert isinstance(cm, pd.DataFrame)
        assert cm.loc['Actual 0', 'Predicted 0'] == 3
        assert cm.loc['Actual 0', 'Predicted 1'] == 1
        assert cm.loc['Actual 1', 'Predicted 0'] == 1
        assert cm.loc['Actual 1', 'Predicted 1'] == 2