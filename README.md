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
- Replace load and save functions with Artifacts class
- Replace Random Forest for Neural Net
- Include plots in output
- Test regression problem with new datasets 
- Create AutoML for classification problems


