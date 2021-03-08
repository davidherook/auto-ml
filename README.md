# Auto-ML

A quick way to train a baseline model. All you need to do is add a dataset (csv file) to the data folder and edit the config file. A trained model is saved to the model folder to serve predictions later and training performance is saved to the output folder.

## Getting Started

Clone the repo, activate a virtual environment, and install requirements:
```
git clone https://github.com/davidherook/auto-ml
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Train a Model

There is a sample config file in config.yaml. Update the features and target for your dataset, specify whether it is a classification or regression problem, and uncomment one of the model types along with its parameters. Pass the config and the training data paths as command line arguments to begin training. A model will be saved to the `model` folder along with the config it was trained with. Training validation results will be written to the `output` folder.

```
# Regression Examples
python train.py -c config.yaml -d data/sample/regression/boston_housing.csv
python train.py -c config.yaml -d data/sample/regression/avocado.csv

# Classification Examples
python train.py -c config.yaml -d data/sample/classification/blobs.csv
python train.py -c config.yaml -d data/sample/classification/circles.csv
```

If you do not want to save any artifacts to `model` or `output`, use the `--no-save` option:
```
python train.py -c config.yaml -d data/sample/classification/circles.csv --no-save
```

### Cross Validation

To ensure that our model will generalize well to unseen test data, there is also the option to run cross validation. This will train/validate the same way as above except over 4 folds of the data rather than one. A test set is excluded from this process and held out for evaluation the same way as above. The persisted model saved to the `model` folder is the one from the last fold. You will notice an additional plot in the `output` folder called `cross_val.png` which gives our model history for each fold. Each line should show a similar trend to show that we're generalizing well to randomness. The training metrics in the top row should be similar to the validation metrics in the bottom row to make sure we're not overfitting.

## Predict 

Identify a model and the path to the test data. Predictions are made using the trained model and the config from the specified hash. Predictions are written to the same directory as the test data with an additional column for the predicted value. The file will be named the same with the suffix "_predictions" added. 
```
python predict.py --model 1614042394 --data data/test_data.csv
```



TODO:
- Preprocessing
- Feature Engineering
- Class Probabilities
- Test with more datasets
- Cross val for scikit
- Test all cases: 
1. Scikit, no cross val 
2. Scikit, cross val 
3. NN, no cross val 
4. NN, cross val 
5. all repeated, with no-save option


