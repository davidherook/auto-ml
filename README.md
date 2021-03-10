# Auto-ML

A quick way to train a baseline model. All you need to do is add your data and edit the config file. A model is saved to the `model` folder and a training performance report is saved to the `output` folder.

## Getting Started

Clone the repo, activate a virtual environment, and install requirements:
```
git clone https://github.com/davidherook/auto-ml
cd auto-ml
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
mkdir model output
```

## Train a Model

In `config.yaml`, update the features and target for your dataset, specify whether it is a classification or regression problem, and uncomment one of the models with its parameters. Pass the config and the training data paths as command line arguments to train:

```
# Regression
python train.py -c data/sample/regression/config_avocado.yaml -d data/sample/regression/avocado.csv

# Classification
python train.py -c data/sample/classification/config_blobs.yaml -d data/sample/classification/blobs.csv
python train.py -c data/sample/classification/config_circles.yaml -d data/sample/classification/circles.csv
```

A model will be saved to the `model` folder along with the config used for training. Training validation results will be written to the `output` folder along with the test set predictions. If you do not wish to save any artifacts, use the `--no-save` option:
```
python train.py -c data/sample/classification/config_circles.yaml -d data/sample/classification/circles.csv --no-save
```

### Training with Cross Validation

To ensure that our model will generalize well to unseen test data, cross validation is available. A test set is excluded from this process and held out for evaluation the same way as above. 

The model saved to the `model` folder is the one from the last fold. Cross validation metrics will be printed for each fold (accuracy for classification and R2 for regression) but if training a neural net, you will notice an additional plot in the `output` folder called `cross_val.png` which gives the model history and accuracy over each fold. Each line should show a similar trend indicating that it's **generalizing well to the random splits**. The training metrics in the top row should be similar to the validation metrics in the bottom row to make sure **the model is not overfitting**.

![alt text](https://github.com/davidherook/auto-ml/blob/master/data/sample/sample_cross_val_output.png?raw=true)

By default, cross validation holds out 20% as test data and then splits over 4 folds. If you have a dataset of 200k rows, the cross validation sets will be:

160k across 4 folds:
1. Train 120k, val 40k
2. Train 120k, val 40k 
3. Train 120k, val 40k 
4. Train 120k, val 40k 

40k remaining for test evaluation

```
python train.py -c data/sample/classification/config_circles.yaml -d data/sample/classification/circles.csv --cross-val
```

After training, you will have model artifacts in the `model` folder including the model and the config it was trained with. In the `output` folder you will have a `metrics.json` and `test_data.csv` along with plots relevant to the problem type.

## Predict 

To make predictions with a trained model, pass the model hash along with the test data. The predictions file will be written to the same directory as the test data with the same name plus the suffix "_predictions":

```
python predict.py --model 1615414293 --data data/sample/regression/avocado_test.csv
```



TODO:
- Preprocessing
- Feature Engineering
- Class Probabilities
- ROC curve 


