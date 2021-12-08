"""
Generic script for visualizing a data set.
"""

# imports for script
from sklearn.decomposition import PCA
from umap import UMAP
from orthrus.core.dataset import DataSet
from orthrus.manifold.mds import MDS
from orthrus.core.helper import module_from_path
from orthrus.core.helper import default_val
from plotly import graph_objs as go
from plotly import offline
import plotly.express as px
import pandas as pd
import numpy as np
from copy import deepcopy

def visualize_timeseries(ds: DataSet, color_attr: str,
                         subject_attr: str, time_attr: str,
                         title:str, dim: int, save_path: str):

    # check if data is continuous or not
    color = ds.metadata[color_attr]
    color_type = pd.api.types.infer_dtype(color.loc[~color.isna()])
    if color_type == 'floating':
        cmax = ds.metadata[color_attr].max()
        cmin = ds.metadata[color_attr].min()
    else:
        color_states = ds.metadata[color_attr].unique()
        colors = px.colors.qualitative.Set1[:color_states.size]
        color_dict = dict(zip(color_states, colors))

    # intialize figure
    fig = go.Figure()

    # create marker dictionary
    times = np.sort(ds.metadata[time_attr].unique().astype(float))
    mrkrs = ['circle']*times.size; mrkrs[0] = 'square'; mrkrs[-1] = 'diamond'
    mrkr_dict = dict(zip(times.astype(str).tolist(), mrkrs))

    # plot each subject
    subject_numbers = ds.metadata[subject_attr].unique()
    subject_numbers.sort()
    for i, subject in enumerate(subject_numbers):
        if i == 0:
            showscale = True
        else:
            showscale = False
        index = ds.metadata[subject_attr] == subject
        df = deepcopy(ds.metadata[index])
        df[time_attr] = df[time_attr].astype(float)
        df.sort_values(by=time_attr, inplace=True)
        df[time_attr] = df[time_attr].astype(str)
        data = deepcopy(ds.data.loc[df.index])
        if dim == 2:
            if color_type != 'floating':
                color_state = df[color_attr].unique()[0]
                fig.add_trace(go.Scatter(x=data.iloc[:, 0],
                                         y=data.iloc[:, 1],
                                         #text=df[time_attr].tolist(),
                                         textposition="top center",
                                         marker=dict(size=8,
                                                     color=color_dict[color_state],
                                                     symbol=df[time_attr].apply(lambda x: mrkr_dict[x]).tolist(),
                                                     ),
                                         mode='lines+markers+text',
                                         line=dict(width=3,
                                                   dash='dash',
                                                   ),
                                         name=str(subject) + "(" + color_state + ")",
                                         visible='legendonly',
                                         )
                              )

                fig.update_layout(legend=dict(orientation="h",
                                              yanchor="bottom",
                                              y=-.2,
                                              xanchor="right",
                                              x=1),
                                  title={
                                      'text': title,
                                      'y': 0.9,
                                      'x': 0.5,
                                      'xanchor': 'center',
                                      'yanchor': 'top'},
                                  scene=dict(xaxis=dict(title=data.columns[0]),
                                             yaxis=dict(title=data.columns[1])),
                                  )
        elif dim == 3:
            if color_type != 'floating':
                color_state = df[color_attr].unique()[0]
                fig.add_trace(go.Scatter3d(x=data.iloc[:, 0],
                                           y=data.iloc[:, 1],
                                           z=data.iloc[:, 2],
                                           # text=df[time_attr].tolist(),
                                           textposition="top center",
                                           marker=dict(size=4,
                                                       color=color_dict[color_state],
                                                       symbol=df[time_attr].apply(lambda x: mrkr_dict[x]).tolist(),
                                                       ),
                                           mode='lines+markers+text',
                                           line=dict(width=3,
                                                     dash='dash',
                                                     ),
                                           name=str(subject) + "(" + color_state + ")",
                                           visible='legendonly',
                                           )
                              )

                fig.update_layout(legend=dict(orientation="h",
                                              yanchor="bottom",
                                              y=-.2,
                                              xanchor="right",
                                              x=1),
                                  title={
                                      'text': title,
                                      'y': 0.9,
                                      'x': 0.5,
                                      'xanchor': 'center',
                                      'yanchor': 'top'},
                                  scene=dict(xaxis=dict(title=data.columns[0]),
                                             yaxis=dict(title=data.columns[1]),
                                             zaxis=dict(title=data.columns[2])),
                                  )

    offline.plot(fig, filename=save_path)

if __name__ == '__main__':
    # imports for arguments
    import argparse
    import os

    # command line arguments
    parser = argparse.ArgumentParser("timeseries-visualization")
    parser.add_argument('--exp_params',
                        type=str,
                        default=os.path.join(os.environ['ORTHRUS_PATH'], 'test_data', 'Iris', 'Experiments',
                                             'classify_setosa_versicolor_svm',
                                             'classify_setosa_versicolor_svm_params.py'),
                        help='File path of containing the experimental parameters. Default is the Iris experiment.')

    parser.add_argument('--embedding',
                        type=str,
                        default='pca',
                        choices=['pca', 'mds', 'umap'],
                        help='Method used to embed the data.')

    parser.add_argument('--dim',
                        type=int,
                        default=2,
                        choices=[1, 2, 3],
                        help='Dimension of embedding.')

    args = parser.parse_args()


    # set experiment parameters
    exp_params = module_from_path('exp_params', args.exp_params)
    script_args = exp_params.VISUALIZE_TIMESERIES_ARGS

    ## required script params
    fig_dir = script_args.get('FIG_DIR', exp_params.FIG_DIR)
    exp_name = script_args.get('EXP_NAME', exp_params.EXP_NAME)
    class_attr = script_args.get('CLASS_ATTR', exp_params.CLASS_ATTR)
    ds = script_args.get('DATASET', exp_params.DATASET)

    ## optional script params
    sample_ids = script_args.get('SAMPLE_IDS', default_val(exp_params, 'SAMPLE_IDS'))
    subject_attr = script_args.get('SUBJECT_ATTR', default_val(exp_params, 'SUBJECT_ATTR'))
    time_attr = script_args.get('TIME_ATTR', default_val(exp_params, 'TIME_ATTR'))
    feature_ids = script_args.get('FEATURE_IDS', default_val(exp_params, 'FEATURE_IDS'))
    save_name = script_args.get('SAVE_NAME', default_val(exp_params, 'VISUALIZE_SAVE_NAME'))
    title = script_args.get('TITLE', default_val(exp_params, 'VISUALIZE_TITLE', val=''))

    # grab dimension
    if args.dim == 1:
        dim = 1
    if args.dim == 2:
        dim = 2
    elif args.dim == 3:
        dim = 3

    # grab embedding
    if args.embedding == 'umap':
        embedding = UMAP(n_components=dim)
    elif args.embedding == 'pca':
        embedding = PCA(n_components=dim, whiten=True, random_state=0)
    elif args.embedding == 'mds':
        embedding = MDS(n_components=dim)

    # custom xlabel and ylabel for PCA
    if args.embedding == 'pca':
        embedding.__str__ = lambda : 'PC()'

    # set save name
    if save_name is None:
        if cross_attr is None:
            save_name = '_'.join([ds.name, exp_name, embedding.__str__().split('(')[0].lower(), class_attr.lower()])
        else:
            save_name = '_'.join([ds.name, exp_name, embedding.__str__().split('(')[0].lower(), class_attr.lower(), cross_attr.lower()])

    # set figure directory just in case
    ds.path = fig_dir

    # slice data
    ds = ds.slice_dataset(sample_ids=sample_ids, feature_ids=feature_ids)

    # compute embedding
    data = pd.DataFrame(index=ds.data.index, data=embedding.fit_transform(ds.data))
    ds.data = data
    ds.data.columns = [embedding.__str__().strip("()") + " " + str(i+1) for i in range(dim)]

    # generate save path
    if save_name is None:
        save_name = '_'.join([self.name, embedding, str(class_attr), str(dim), "timeseries"])
    save_path = os.path.join(ds.path, save_name + ".html")

    # visualize
    visualize_timeseries(ds=ds, color_attr=class_attr,
                         time_attr=time_attr, subject_attr=subject_attr,
                         title=title, dim=dim, save_path=save_path)


