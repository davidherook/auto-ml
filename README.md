# Auto-ML

A quick way to train a baseline model.

## Getting Started

Clone the repo, install requirements, and activate the virtual environment
```
git clone ...
pip install -r requirements.txt
source venv/bin/activate
```

## Train a Model

Identify the training data and the config file
```
python train.py -c config.yaml -d data/boston_housing.csv
```



TODO:
- Fix settingwithcopywarning in auto_ml line 35, 36
- Put model and conf in same dir
- Replace load and save functions for Artifacts


