"""
This script visualizes the Iris data set.
"""

if __name__ == '__main__':

    # imports
    import sys
    from sklearn.decomposition import PCA
    from umap import UMAP
    from datasci.manifold.mds import MDS

    # set experiment parameters
    from Experiments.setosa_versicolor_classify_species_svm import setosa_versicolor_classify_species_svm_params as exp_params
    exp_name = exp_params.EXP_NAME
    class_attr = exp_params.CLASS_ATTR
    ds = exp_params.DATASET
    sample_ids = exp_params.SAMPLE_IDS

    # visualize data
    umap = UMAP(n_components=2, n_neighbors=25)
    pca = PCA(n_components=2, whiten=True, random_state=0)
    mds = MDS(n_components=2)
    embedding = pca # <---- Choose one
    backend = 'pyplot'
    figsize_plotly = (1500, 1000)
    palette_pyplot = 'bright'
    ds.visualize(embedding=embedding,
                 sample_ids=sample_ids,
                 attr=class_attr,
                 backend=backend,
                 palette='bright',
                 alpha=.7,
                 subtitle='', # <--- default show normalization and imputation methods used
                 s=200,
                 save=True,
                 save_name='_'.join([ds.name, exp_name, embedding.__str__().split('(')[0].lower(), class_attr]))