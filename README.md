# Out of the Box Machine Learning

This project was created to have a model available immediately which is ready to supply predictions on new data. The user only needs to add their dataset to the `data` folder and configure the `config.yaml` file.

The process provides: 
- validation metrics for each model
- entire dataset along with test set predictions from each model
- pickled model files that are ready to use for predictions

The following types of problems are supported:
- regression 
- binary classification

The following files are written after training:
1. output/[latest_model_run]results.csv
1. output/[latest_model_run]validation_metrics.....
2. tableau/results.csv
3. tableau/column_alias_names.json
4. model/....

## Choose Problem Type
If regression problem:
```
cd regression
```

If classification:
```
cd classification
```

## Train a model 

```
# Train
python train.py -c config.yaml

# Train and Persist Models
python train.py -c config.yaml --save_output --save_model
```

## Predict in Batch
```
python predict.py -c config.yaml -d data/boston_housing.csv -m model/rf_regression.pk
```



TODO:
- configure models in config file
- scaling
- resampling
- combine plots into one fig
- write predictions to separate folder
- write readme
- dockerize
- k-fold validation