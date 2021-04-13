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
                        choices=[2, 3],
                        help='Dimension of embedding.')

    args = parser.parse_args()

    # imports for script
    from sklearn.decomposition import PCA
    from umap import UMAP
    from datasci.manifold.mds import MDS
    from datasci.core.helper import module_from_path

    # set experiment parameters
    exp_params = module_from_path('exp_params', args.exp_params)
    exp_name = exp_params.EXP_NAME
    class_attr = exp_params.CLASS_ATTR
    ds = exp_params.DATASET
    sample_ids = exp_params.SAMPLE_IDS

    # grab dimension
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

    # visualize data
    backend = 'pyplot'
    figsize_plotly = (1500, 1000)
    palette_pyplot = 'bright'
    ds.visualize(embedding=embedding,
                 sample_ids=sample_ids,
                 attr=class_attr,
                 backend=backend,
                 palette='bright',
                 alpha=.7,
                 mrkr_list=['o'],
                 subtitle='', # <--- default show normalization and imputation methods used
                 s=200,
                 save=True,
                 save_name='_'.join([ds.name, exp_name, embedding.__str__().split('(')[0].lower(), class_attr]))
