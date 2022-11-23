import os
import pandas as pd
import numpy as np

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from orthrus.sparse.feature_selection.IterativeFeatureRemoval import IFR
from orthrus.core.pipeline import Pipeline
from orthrus.core.helper import load_object, save_object


import logging
logger = logging.getLogger(__name__)


def process_ifr_tuning_results(**kwargs):
    results_location = kwargs['results_location']
    tune_results = kwargs['tune_results']
    logger.info(f'Processing results for IFR execution now.')


    main_figure = make_subplots(rows=1, cols=2, subplot_titles=("Mean Training Score", "Mean Validation Score"))
    iterations_scatter_fig = make_subplots(rows=1, cols=1)

    for i, (index, row) in enumerate(tune_results.iterrows()):
        try:
            pipeline: Pipeline = load_object(os.path.join(row['logdir'],  'pipeline.pkl'))
        except FileNotFoundError:
            logger.error(f'Pipeline not found at location {results_location}. Skipping the pipeline for processing the results.')
            continue



        # log feature count metric and train-validation scores
        train_score_means = pd.DataFrame()
        train_score_stds = pd.DataFrame()

        validation_score_means = pd.DataFrame()
        validation_score_stds = pd.DataFrame()

        score_presence_fraction = pd.DataFrame()

        fc_metrics = pd.DataFrame()

        for batch, pipeline_results in pipeline.results_.items():
            ifr:IFR = pipeline_results['selector']

            # get mean scores
            train_s_mean = ifr.diagnostic_information_['train_scores'].mean(axis=1).values
            validation_s_mean = ifr.diagnostic_information_['validation_scores'].mean(axis=1).values

            # get std deviation
            train_s_std = ifr.diagnostic_information_['train_scores'].std(axis=1).values
            validation_s_std = ifr.diagnostic_information_['validation_scores'].std(axis=1).values

            # get fraction of values present (this will be used for marker size)
            values_presence_fraction = (1 - ifr.diagnostic_information_['train_scores'].isna().sum(axis=1) / ifr.diagnostic_information_['train_scores'].shape[1]) * 25


            # append to dataframes
            train_score_means = pd.concat([train_score_means, pd.DataFrame(train_s_mean, columns=[batch], index=np.arange(train_s_mean.shape[0]))], axis=1)
            train_score_stds = pd.concat([train_score_stds, pd.DataFrame(train_s_std, columns=[batch], index=np.arange(train_s_std.shape[0]))], axis=1)
            
            validation_score_means = pd.concat([validation_score_means, pd.DataFrame(validation_s_mean, columns=[batch], index=np.arange(validation_s_mean.shape[0]))], axis=1)
            validation_score_stds = pd.concat([validation_score_stds, pd.DataFrame(validation_s_std, columns=[batch], index=np.arange(validation_s_std.shape[0]))], axis=1)
            
            score_presence_fraction = pd.concat([score_presence_fraction, pd.DataFrame(values_presence_fraction, columns=[batch], index=np.arange(values_presence_fraction.shape[0]))], axis=1)

            feature_counts = ifr.diagnostic_information_['true_feature_count']
            total_feautres_extracted = feature_counts.sum().sum()
            total_iterations = feature_counts.count().sum()

            fc_metrics = pd.concat([fc_metrics, pd.DataFrame([[row['config/ssvm_C'], total_iterations, total_feautres_extracted/(total_iterations + 1e-7)]], columns=['C', 'Total iterations', 'Avg features per iteration'])], axis=0)

        # ------------------------------
        # Plot # 1
        # ------------------------------


        train_score_means = pd.DataFrame(train_score_means.mean(axis=1).values, columns=['score']) 
        train_score_stds = train_score_stds.mean(axis=1)

        validation_score_means = pd.DataFrame(validation_score_means.mean(axis=1).values, columns=['score']) 
        validation_score_stds = validation_score_stds.mean(axis=1)
                    
        score_presence_fraction = score_presence_fraction.mean(axis=1)


        main_figure.add_trace(go.Scatter(x=train_score_means.index, 
                                            y=train_score_means['score'], 
                                            marker_size=score_presence_fraction.values,
                                            error_y=dict(type='data', 
                                                        array=train_score_stds.values,
                                                        visible=True),
                                            line=dict(
                                                width= 6 if i == 0 else 2),
                                            mode="lines+markers",
                                            name=f'C: {row["config/ssvm_C"]}'), 
                            row=1, 
                            col=1)

        main_figure.add_trace(go.Scatter(x=validation_score_means.index, 
                                            y=validation_score_means['score'], 
                                            marker_size=score_presence_fraction.values,
                                            error_y=dict(type='data', 
                                                        array=validation_score_stds.values,
                                                        visible=True),
                                            line=dict(
                                                width= 6 if i == 0 else 2),
                                            mode="lines+markers",
                                            name=f'C: {row["config/ssvm_C"]}'), 
                            row=1, 
                            col=2)

        main_figure.update_layout(
            title="Iteration Number vs Score for C parameter",
            xaxis_title="Iteration",
            yaxis_title="Score",
            legend_title="SSVM C Values",
            font=dict(
                # family="Courier New, monospace",
                size=18,
                color="RebeccaPurple"
            )
        )



        # ------------------------------
        # Plot # 2
        # ------------------------------



        fc_mean = fc_metrics.mean()
        iterations_scatter_fig.add_trace(go.Scatter(x = [fc_mean["Total iterations"]],
                                                    y = [fc_mean["Avg features per iteration"]],
                                                    mode="markers",
                                                    marker_size=18 if i==0 else 8,
                                                    name=f'C: {row["config/ssvm_C"]}'), 
                                        row=1, 
                                        col=1)
        iterations_scatter_fig.update_layout(
            title='Total iterations vs Average features per iteration',
            xaxis_title="Total iterations",
            yaxis_title="Average features per iteration",
            legend_title="SSVM C Values",
            font=dict(
                # family="Courier New, monospace",
                size=18,
                color="RebeccaPurple"
            )
        )

    figures_dirname = 'figures'
    os.makedirs(os.path.join(results_location, figures_dirname), exist_ok=True)
    fname = 'ifr_ssvm_c_tuning_iterations_scatter'
    iterations_scatter_fig.write_html(os.path.join(results_location, figures_dirname, f'{fname}.html'))


    fname = 'ifr_ssvm_c_tuning_training_and_validation_scores'
    main_figure.write_html(os.path.join(results_location, figures_dirname, f'{fname}.html'))



def process_ifr_results(**kwargs):

    results_location = kwargs['results_location']
    pipeline = kwargs['pipeline']

    csv_location = os.path.join(results_location, 'ifr_feature_sets', 'csv')
    plot_location = os.path.join(results_location, 'ifr_feature_sets', 'plots')
    os.makedirs(csv_location, exist_ok=True)
    os.makedirs(plot_location, exist_ok=True)

    all_features = pd.DataFrame(columns=['batch_id', 'features'])
    all_features.set_index('batch_id', inplace=True)

    summary = pd.DataFrame(columns=['batch_id', 'Number of features'])
    summary.set_index('batch_id', inplace=True)

    for batch_id, results in pipeline.results_.items():
        f_ranks = results['f_ranks']
        f_ranks = f_ranks.loc[f_ranks['frequency'] > 0]
        f_ranks.to_csv(os.path.join(csv_location, f'{batch_id}_features.csv'))

        plot_feature_frequency(f_ranks, os.path.join(plot_location, f'{batch_id}.png'))

        df = pd.DataFrame([[batch_id, f_ranks.index.values]], columns=['batch_id', 'features'])
        df.set_index('batch_id', inplace=True)
        all_features= pd.concat([all_features, df])


        df = pd.DataFrame([[batch_id, len(f_ranks)]], columns=['batch_id', 'Number of features'])
        df.set_index('batch_id', inplace=True)
        summary= pd.concat([summary, df])

    save_object(all_features,os.path.join(csv_location, f'all_features.pkl'))
    all_features.to_csv(os.path.join(csv_location, f'all_features.csv'))
    summary.to_csv(os.path.join(csv_location, f'ifr_feature_counts.csv'))

def plot_feature_frequency(f_ranks, path, attr='frequency'):
    import matplotlib.pyplot as plt
    from orthrus.sparse.feature_selection.helper import rank_features_by_attribute
    ranked_feature_ids = rank_features_by_attribute(f_ranks, {'attr': attr, 'order': 'desc'})
    f_ranks = f_ranks.loc[ranked_feature_ids]
    f_ranks = f_ranks.loc[f_ranks[attr] > 0]
    fig, axs = plt.subplots(1,1)
    axs.plot(np.arange(len(f_ranks)), f_ranks[attr].values)
    axs.set_ylabel("Frequency")
    axs.set_xlabel("Feature index")
    # axs.set_title(labels[tranfrom_id])
    plt.savefig(path)     
    
def process_rfs_results(**kwargs):
    results_location = kwargs['results_location']
    pipeline = kwargs['pipeline']

    csv_location = os.path.join(results_location, 'rfs_feature_sets', 'csv')
    plot_location = os.path.join(results_location, 'rfs_feature_sets', 'plots')
    os.makedirs(csv_location, exist_ok=True)
    os.makedirs(plot_location, exist_ok=True)

    summary = pd.DataFrame(columns=['batch_id', 'Number of features'])
    summary.set_index('batch_id', inplace=True)

    all_features = pd.DataFrame(columns=['batch_id', 'features'])
    all_features.set_index('batch_id', inplace=True)

    for batch_id, results in pipeline.results_.items():
        features = results['reduced_feature_ids']
        n_features = len(features) 
        df = results['optimal_n_results'].sort_values('size')
        
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1,1)
        axs.plot(df.index, df['score'])
        axs.plot(n_features, df.loc[n_features,'score'], 'ro')
        axs.set_ylabel("Score")
        axs.set_xlabel("Num Features")
        # axs.set_title(labels[tranfrom_id])
        plt.savefig(os.path.join(plot_location, f'{batch_id}.png'))


        df = pd.DataFrame([[batch_id, features]], columns=['batch_id', 'features'])
        df.set_index('batch_id', inplace=True)
        all_features= pd.concat([all_features, df])

        df = pd.DataFrame([[batch_id, len(features)]], columns=['batch_id', 'Number of features'])
        df.set_index('batch_id', inplace=True)
        summary= pd.concat([summary, df])

    save_object(all_features,os.path.join(csv_location, f'all_features.pkl'))
    all_features.to_csv(os.path.join(csv_location, f'all_features.csv'))
    summary.to_csv(os.path.join(csv_location, f'rfs_feature_counts.csv'))

def process_classification_results(**kwargs):
    pipeline = kwargs['pipeline']
    from orthrus.core.pipeline import Report
    import orthrus.mlflow.modules.utils as utils
    report: Report = pipeline.processes[-1]
    utils.log_report_scores(report)

def process_svm_c_tuning_results(**kwargs):
        import matplotlib.pyplot as plt
        
        fig, axs = plt.subplots(1,1)
        svm_c = kwargs['tune_results']['config/svm_C'].values
        valid_bsrs = kwargs['tune_results']['mean_valid_bsr'].values

        axs.scatter(svm_c[1:], valid_bsrs[1:])
        axs.scatter(svm_c[0], valid_bsrs[0], color='green', label='best run')
        axs.set_xlabel("SVM C parameter value")
        axs.set_ylabel("Mean validation BSR across batches")
        plt.legend()
        plt.savefig(os.path.join(kwargs['results_location'], 'scatter_plot.png'))