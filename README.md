# Out of the Box Machine Learning

The purpose of this project is to provide a trained machine learning model, for any new dataset, without rewriting code. The user needs only to fill out the config file and provide the training data. 

The process provides: 
- validation metrics for each model
- entire dataset along with test set predictions from each model
- pickled model files that are ready to use for predictions


The following types of problems are supported:
- regression 
- classification

The following files are written after training:
1. output/[latest_model_run]results.csv
1. output/[latest_model_run]validation_metrics.....
2. tableau/results.csv
3. tableau/column_alias_names.json
4. model/....

## Train a model 

```
python regression/train.py -c regression/config.yaml
```

#### TODO

- create tableau validation notebook
- get working on real data set
- include neural net