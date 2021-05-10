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
                                             'setosa_versicolor_classify_species_svm',
                                             'setosa_versicolor_classify_species_svm_params.py'),
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
    import pandas as pd

    # set experiment parameters
    exp_params = module_from_path('exp_params', args.exp_params)
    exp_name = exp_params.EXP_NAME
    class_attr = exp_params.CLASS_ATTR
    ds = exp_params.DATASET
    sample_ids = exp_params.SAMPLE_IDS
    feature_ids = exp_params.FEATURE_IDS

    # grab dimension
    if args.dim == 1:
        dim = 1
    if args.dim == 2:
        dim = 2
    elif args.dim == 3:
        dim = 3

    # grab embedding
    if args.embedding == 'umap':
        embedding = UMAP(n_components=dim, n_neighbors=30)
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
        backend_args = dict(palette=palette, alpha=.7, mrkr_list=['o'], s=200)
    elif args.backend == 'plotly':
        backend = 'plotly'
        backend_args = dict(figsize=(1500, 1000))

    # visualize data
    ds.visualize(embedding=embedding,
                 sample_ids=sample_ids,
                 feature_ids=feature_ids,
                 attr=class_attr,
                 backend=backend,
                 subtitle='', # <--- default show normalization and imputation methods used
                 save=True,
                 save_name='_'.join([ds.name, exp_name, embedding.__str__().split('(')[0].lower(), class_attr.lower()]),
                 **backend_args)
