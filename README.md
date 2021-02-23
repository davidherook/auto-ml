# Auto-ML

A quick way to train a baseline model.

## Getting Started

Clone the repo, install requirements, and activate the virtual environment
```
git clone https://github.com/davidherook/auto-ml
pip install -r requirements.txt
source venv/bin/activate
```

## Train a Model

Identify the training data and the config file
```
python train.py -c config.yaml -d data/boston_housing.csv
```

## Predict 

Identify a model and the path to the test data. Predictions are written to the same directory with the suffix "_predictions"
```
python predict.py --model 1614042394 --data data/test_data.csv
```



TODO:
- Fix settingwithcopywarning in auto_ml line 35, 36
- Replace load and save functions for Artifacts
- Replace Random Forest for Neural Net
- Write model performance analyzer


