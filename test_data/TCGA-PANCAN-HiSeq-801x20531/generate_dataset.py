"""
This script generates a dataset object for the TCGA-PANCAN-HiSeq-801x20531 cancer RNA-seq
downlaoded from http://archive.ics.uci.edu/ml/datasets/gene+expression+cancer+RNA-Seq#
"""

if __name__ == "__main__":
    # imports
    from orthrus.core.dataset import DataSet
    from orthrus.preprocessing.imputation import HalfMinimum
    import os
    import pandas as pd
    import numpy as np

    # load data and metadata
    data = pd.read_csv(os.path.join(os.environ['ORTHRUS_PATH'], "test_data\\TCGA-PANCAN-HiSeq-801x20531\\Data\\data.csv"),
                       index_col=0)
    metadata = pd.read_csv(os.path.join(os.environ['ORTHRUS_PATH'],"test_data\\TCGA-PANCAN-HiSeq-801x20531\\Data\\labels.csv"),
                           index_col=0).rename(columns={'Class': 'tumor_class'})
    description = "This collection of data is part of the RNA-Seq (HiSeq) PANCAN data set, it is a random extraction of gene " \
                  "expressions of patients having different types of tumor: BRCA, KIRC, COAD, LUAD and PRAD."

    # Only keep genes which are expressed highly in at least one group
    fids = pd.Series(index=data.columns, data=[False]*data.shape[1])
    for tumor_class in metadata['tumor_class'].unique():
        sids = metadata['tumor_class'] == tumor_class
        expression_prop = (data.loc[sids] != 0).sum(axis=0) / sids.sum()
        fids = fids | (expression_prop > .8)
    data = data.loc[:, fids]

    # impute half-minimum and log2 transform
    data = pd.DataFrame(data=HalfMinimum(missing_values=0).fit_transform(data), index=data.index, columns=data.columns)
    data = pd.DataFrame(data=np.log2(data), index=data.index, columns=data.columns)

    # create DataSet object
    ds = DataSet(name="TCGA-PANCAN-HiSeq-801x20531-log2", data=data, metadata=metadata, description=description)

    # save dataset object
    ds.save(os.path.join(os.environ['ORTHRUS_PATH'],
                         "test_data\\TCGA-PANCAN-HiSeq-801x20531\\Data\\TCGA-PANCAN-HiSeq-801x20531-log2.ds"),
                         overwrite=True)
