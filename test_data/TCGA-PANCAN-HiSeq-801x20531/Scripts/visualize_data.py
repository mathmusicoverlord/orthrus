"""This script is to provide an example pipeline for new users."""


if __name__ == "__main__":
    # imports
    import os
    from sklearn.decomposition import PCA
    from orthrus.core.dataset import load_dataset

    # load dataset
    ds = load_dataset(os.path.join(os.environ['ORTHRUS_PATH'],
                                   "test_data\\TCGA-PANCAN-HiSeq-801x20531\\Data\\TCGA-PANCAN-HiSeq-801x20531-log2.ds"))
    ds.path = os.path.join(os.environ["ORTHRUS_PATH"], "docsrc/figures")

    # visualize
    pca = PCA(whiten=True, n_components=3)
    ds.visualize(embedding=pca,
                 backend='plotly',
                 attr='tumor_class',
                 save=True,
                 save_name='TCGA-PANCAN-HiSeq-801x20531_pca_viz_example_4_3d',
                 figsize=(1500, 1000),
                 mrkr_size=10,
                 opacity=.7,
                 subtitle='')