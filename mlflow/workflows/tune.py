"""Script for tuning an orthrus pipeline."""


# imports
import argparse
from orthrus.core.helper import module_from_path
from orthrus.core.dataset import load_dataset
from orthrus.core.dataset import DataSet
import ray.tune
from  mlflow.tracking import MlflowClient
from ray.tune.integration.mlflow import mlflow_mixin
from orthrus.core.pipeline import Pipeline, Report
from modules import utils
import yaml
import os
import mlflow

# command line arguments
parser = argparse.ArgumentParser("Tune", description="Tunes a pipeline on a dataset using ray.tune.")
parser.add_argument("--dataset-path", dest="dataset_path",
                    type=str, help="File path to the dataset")
parser.add_argument("--sample-query", dest="sample_query",
                    type=str, help="String query passed to DataSet.metadata.query() for filtering samples.")
parser.add_argument("--pipeline-path", dest="pipeline_path",
                    type=str, help="File path to the pipeline python file. This module must contain the "\
                                    "methods: config(), search_alg(), scheduler(), score(), and generate_pipeline(). See "\
                                    "the example pipeline under orthrus/pipelines/svm.py")
# parser.add_argument("--experiment-name", dest="experiment_name",
#                     type=str, help="Name of the MLFlow experiment.")
parser.add_argument("--num-samples", dest="num_samples",
                    type=int, help="Number of hyperparameters to sample.")

args = parser.parse_args()

# load the pipeline module
pipeline_name = os.path.basename(args.pipeline_path).rstrip('.py')
pipeline_module = module_from_path(pipeline_name, args.pipeline_path)

# extract experiment id
experiment_id = os.environ['MLFLOW_EXPERIMENT_ID']

# user defined functions
@mlflow_mixin
def set_description():
    mlflow.set_tag('mlflow.note.content',
                f"Tuning of pipeline {os.path.basename(args.pipeline_path)} on dataset {os.path.basename(args.dataset_path)} "\
                f"restricted to samples queried by: {args.sample_query}. {args.num_samples} points were sampled in hyperparameter tuning.")

def generate_dataset() -> DataSet:
    # load the dataset
    ds: DataSet = load_dataset(args.dataset_path)

    # slice the dataset
    sample_ids = ds.metadata.query(args.sample_query).index
    ds = ds.slice_dataset(sample_ids=sample_ids)

    # convert metadata to string type
    ds.metadata = ds.metadata.astype(str)

    return ds

@mlflow_mixin
def trainable(config: dict) -> dict:
    # generate the dataset
    ds = generate_dataset()

    # generate pipeline module
    pipeline_name = os.path.basename(args.pipeline_path).rstrip('.py')
    pipeline_module = module_from_path(pipeline_name, args.pipeline_path)

    # generate pipeline from config
    pipeline: Pipeline = pipeline_module.generate_pipeline(**config)

    # run the pipeline on the data
    pipeline.run(ds)

    # return score
    score = pipeline_module.score(pipeline)
    mlflow.log_metric(key="score", value=score)
    ray.tune.report(score=score)

@mlflow_mixin
def log_config(config_name: str, config: dict):

    # set temp file path
    temp_path = f'tmp/{config_name}.yml'

    # drop mlflow key
    del config['mlflow']

    # save yaml file
    with open(temp_path, 'w+') as config_file:
        yaml.dump(config, config_file, default_flow_style=False)

        # log the artifact
        mlflow.log_artifact(temp_path)

def main():

    # extract config and search algorithm
    config = pipeline_module.config()
    search_alg = pipeline_module.search_alg()
    scheduler = pipeline_module.scheduler()

    # update config with mlflow
    tracking_uri = mlflow.get_tracking_uri()
    config['mlflow'] = {'experiment_id': experiment_id,
                        'tracking_uri': tracking_uri}

    # start tuning
    analysis = ray.tune.run(
        trainable,
        config=config,
        search_alg=search_alg,
        scheduler=scheduler,
        num_samples=args.num_samples,
        fail_fast="raise",
        )
    
    # log best hyperparameters
    best_config = analysis.get_best_config(mode=search_alg.mode, metric=search_alg.metric)
    log_config('best_config', best_config)

    # set description
    set_description()

if __name__ == "__main__":

    main()
