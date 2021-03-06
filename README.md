# Auto-ML

A quick way to train a baseline model. All you need to do is add a dataset to the data folder and choose which features to use in the config file. The model is persisted in the model folder and the training performance is reported in the output folder.

## Getting Started

Clone the repo, install requirements, and activate the virtual environment
```
git clone https://github.com/davidherook/auto-ml
pip install -r requirements.txt
source venv/bin/activate
```

## Train a Model

There is a sample config file in config.yaml. You only need to change the features and target and then specify whether it is a regression or classification problem. To choose a model, uncomment one of the model types along with its parameters. Then, pass the config and the training data path as arguments to train a model. A model will be saved to the model folder along with the config it was trained with. Training validation results will be written to the output folder.

```
# Regression Examples
python train.py -c config.yaml -d data/sample/regression/boston_housing.csv
python train.py -c config.yaml -d data/sample/regression/avocado.csv

# Classification Examples
python train.py -c config.yaml -d data/sample/classification/blobs.csv
python train.py -c config.yaml -d data/sample/classification/circles.csv
```

If you do not wish to persist the trained model and the validation results, use the `--no-save` option:
```
python train.py -c config.yaml -d data/sample/classification/circles.csv --no-save
```

## Predict 

Identify a model and the path to the test data. Predictions are made using the trained model and the config from the specified hash. Predictions are written to the same directory as the test data with an additional column for the predicted value. The file will be named the same with the suffix "_predictions" added. 
```
python predict.py --model 1614042394 --data data/test_data.csv
```



TODO:
- Preprocessing
- Class Probabilities
- Replace validation_split with validation_data
- Type Hints?


