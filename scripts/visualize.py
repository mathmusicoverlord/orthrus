"""
Generic script for visualizing a data set.
"""

if __name__ == '__main__':
    # imports for arguments
    import argparse
    import os

    # command line arguments
    parser = argparse.ArgumentParser("generic-visualization")
    parser.add_argument('--exp_params',
                        type=str,
                        default=os.path.join(os.environ['DATASCI_PATH'], 'test_data', 'Iris', 'Experiments',
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

    parser.add_argument('--backend',
                        type=str,
                        default='pyplot',
                        choices=['plotly', 'pyplot'],
                        help='Dimension of embedding.')

    args = parser.parse_args()

    # imports for script
    from sklearn.decomposition import PCA
    from umap import UMAP
    from datasci.manifold.mds import MDS
    from datasci.core.helper import module_from_path
    from datasci.core.helper import default_val
    import pandas as pd

    # set experiment parameters
    exp_params = module_from_path('exp_params', args.exp_params)
    script_args = exp_params.VISUALIZE_ARGS

    ## required script params
    fig_dir = script_args.get('FIG_DIR', exp_params.FIG_DIR)
    exp_name = script_args.get('EXP_NAME', exp_params.EXP_NAME)
    class_attr = script_args.get('CLASS_ATTR', exp_params.CLASS_ATTR)
    ds = script_args.get('DATASET', exp_params.DATASET)

    ## optional script params
    sample_ids = script_args.get('SAMPLE_IDS', default_val(exp_params, 'SAMPLE_IDS')),
    feature_ids = script_args.get('FEATURE_IDS', default_val(exp_params, 'FEATURE_IDS'))
    cross_attr = script_args.get('CROSS_ATTR', default_val(exp_params, 'CROSS_ATTR'))
    save_name = script_args.get('SAVE_NAME', default_val(exp_params, 'VISUALIZE_SAVE_NAME'))
    title = script_args.get('TITLE', default_val(exp_params, 'VISUALIZE_TITLE', val=''))
    subtitle = script_args.get('SUBTITLE', default_val(exp_params, 'VISUALIZE_SUBTITLE', val=''))

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

    # get backend info
    if args.backend == 'pyplot':
        backend = 'pyplot'
        if pd.api.types.infer_dtype(ds.metadata[class_attr]) == 'floating':
            palette = 'viridis'
        else:
            palette = 'bright'
        if cross_attr is None:
            backend_args = dict(palette=palette, alpha=.7, mrkr_list=['o'], s=200)
        else:
            backend_args = dict(palette=palette, alpha=.7, s=200)
    elif args.backend == 'plotly':
        backend = 'plotly'
        backend_args = dict(figsize=(1500, 1000))

    # custom xlabel and ylabel for PCA
    if args.embedding == 'pca':
        xlabel = 'PC 1'
        if dim > 1:
            ylabel = 'PC 2'
            if dim > 2:
                zlabel = 'PC 3'
            else:
                zlabel = None
        else:
            ylabel = None
    else:
        xlabel = ylabel = zlabel = None

    # set save name
    if save_name is None:
        if cross_attr is None:
            save_name = '_'.join([ds.name, exp_name, embedding.__str__().split('(')[0].lower(), class_attr.lower()])
        else:
            save_name = '_'.join([ds.name, exp_name, embedding.__str__().split('(')[0].lower(), class_attr.lower(), cross_attr.lower()])

    # set figure directory just in case
    ds.path = fig_dir

    # visualize data
    ds.visualize(embedding=embedding,
                 sample_ids=sample_ids,
                 feature_ids=feature_ids,
                 attr=class_attr,
                 cross_attr=cross_attr,
                 backend=backend,
                 subtitle=subtitle, # <--- default show normalization and imputation methods used
                 title=title,
                 save=True,
                 save_name=save_name,
                 **backend_args)
