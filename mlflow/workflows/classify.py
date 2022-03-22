"""Script for MLFlow usage. Generic classification experiment using a classification pipeline."""


# imports
import argparse
from orthrus.core.helper import module_from_path
from orthrus.core.dataset import load_dataset
from orthrus.core.dataset import DataSet
from orthrus.core.pipeline import Pipeline, Report
from modules import utils
import os
import mlflow

# command line arguments
parser = argparse.ArgumentParser("Classify", description="Runs a classification experiment with a dataset and a pre-defined pipeline.")
parser.add_argument("--dataset-path", dest="dataset_path",
                    type=str, help="File path to the dataset")
parser.add_argument("--sample-query", dest="sample_query",
                    type=str, help="String query passed to DataSet.metadata.query() for filtering samples.")
parser.add_argument("--pipeline-path", dest="pipeline_path",
                    type=str, help="File path to the pipeline python file. This module must contain the "\
                                    "variables PIPELINE and HYPERPARAMS, which contain the pipeline to be run "\
                                    "and the hyperparameters of the pipeline, respectively. The pipeline should "\
                                    "terminate with a orthrus.core.pipeline.Report class to maintain consistency "\
                                    "among classification metrics.")
args = parser.parse_args()

# user defined functions
def set_description():
    mlflow.set_tag('mlflow.note.content',
                f"Classification results of pipeline {os.path.basename(args.pipeline_path)} on dataset {os.path.basename(args.dataset_path)} "\
                f"restricted to samples queried by: {args.sample_query}.")

if __name__=="__main__":

    # load the dataset
    ds: DataSet = load_dataset(args.dataset_path)

    # slice the dataset
    sample_ids = ds.metadata.query(args.sample_query).index
    ds = ds.slice_dataset(sample_ids=sample_ids)
    ds.metadata = ds.metadata.astype(str)

    # start mlflow run
    with mlflow.start_run():
        # load the pipeline module
        pipeline_name = os.path.basename(args.pipeline_path).rstrip('.py')
        pipeline_module = module_from_path(pipeline_name, args.pipeline_path)

        # extract hyperparams from pipeline
        hyperparams: dict = pipeline_module.hyperparams()
        os.system("rm -f /tmp/pipeline.pickle")
        pipeline: Pipeline = pipeline_module.generate_pipeline(**hyperparams, checkpoint_path='/tmp/pipeline.pickle')

        # log the parameters
        mlflow.log_params(hyperparams)

        # run the pipeline on the data
        pipeline.run(ds, checkpoint=True)

        # extract classification metrics
        report: Report = pipeline.processes[-1]
        utils.log_report_scores(report)

        # save pipeline artifact
        mlflow.log_artifact(pipeline.checkpoint_path)
    
        # set description
        set_description()