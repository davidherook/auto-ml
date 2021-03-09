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
python train.py -c config.yaml -d data/sample/regression/avocado.csv
python train.py -c config.yaml -d data/sample/regression/boston_housing.csv

# Classification Examples
python train.py -c config.yaml -d data/sample/classification/blobs.csv
python train.py -c config.yaml -d data/sample/classification/circles.csv

# If you do not want to save model artifacts or output
python train.py -c config.yaml -d data/sample/classification/circles.csv --no-save
```

### Training with Cross Validation

To ensure that our model will generalize well to unseen test data, cross validation is available. This will train/validate the same way as above except over 4 folds of the data. A test set is excluded from this process and held out for evaluation the same way as above. 

The model saved to the `model` folder is the one from the last fold. Cross validation metrics will be printed to the screen (accuracy for classification and R2 for regression) but if training a neural net, you will notice an additional plot in the `output` folder called `cross_val.png` which gives the model history and accuracy over each fold. Each line should show a similar trend indicating that it's **generalizing well to the random splits**. The training metrics in the top row should be similar to the validation metrics in the bottom row to make sure **the model is not overfitting**.

![alt text](https://github.com/davidherook/auto-ml/blob/master/data/sample/cross-val.png?raw=true)


![Screenshot](data/sample/cross-val.png)

If not using cross validation, you can set the train/validation/test splits in config. By default, cross validation holds out 20% as test data and then splits over 4 folds. Therefore, if you have a dataset of 200k rows, the cross validation sets will be:

160k across 4 folds:
1. Train 120k, val 40k
2. Train 120k, val 40k 
3. Train 120k, val 40k 
4. Train 120k, val 40k 

40k remaining for test evaluation

```
python train.py -c config.yaml -d data/sample/classification/circles.csv --cross-val
```

After training you should have model artifacts in the `model` folder including the model and the config it was trained with. In the `output` folder you will have a `metrics.json` and `test_data.csv`. There will also be plots that differ whether it is a regression problem (pred vs. actual, residual plot) or a classification problem  (...).  

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
- Cross val for neural net regression problems
- include training on, validating on, testing on... message for cross validation
- move sample configs to sample folders


