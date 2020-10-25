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

## Train a model 

```
# Train
python regression/train.py -c sample_configs/boston_housing.yaml

# Train and Persist Models
python regression/train.py -c sample_configs/boston_housing.yaml --save_models --save_output
```

## Predict in Batch
```
python regression/predict.py -c sample_configs/boston_housing.yaml -d regression/data/boston_housing.csv -m regression/model/rf_regression.pk
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