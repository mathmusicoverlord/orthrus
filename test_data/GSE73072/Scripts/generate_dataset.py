"""
Script to generate the GSE73072 dataset.
"""
if __name__ == "__main__":
    # imports
    import GEOparse
    import os
    import pandas as pd
    from datasci.core.dataset import DataSet

    # load data from geo
    data_dir = os.path.join(os.environ["DATASCI_PATH"], 'test_data/GSE73072/Data')
    gse = GEOparse.get_GEO(geo="GSE73072", destdir=data_dir)

    # extract metadata
    metadata = gse.phenotype_data

    # format the metadata
    metadata.columns = metadata.columns.str.replace('characteristics_ch1.\d.', '')
    metadata.columns = metadata.columns.str.replace('_ch1', '')
    metadata.columns = metadata.columns.str.replace('/', '_')
    metadata.columns = metadata.columns.str.replace(' ', '_')
    metadata.columns = metadata.columns.str.replace('time_point', 'time_point_hr')
    metadata['time_point_hr'] = metadata['time_point_hr'].str.replace('hour ', '').astype(int)

    # extract vardata
    vardata = gse.gpls['GPL14604'].table
    vardata.columns = vardata.columns.str.replace('ID', 'ID_REF')
    vardata.set_index('ID_REF', drop=True, inplace=True)

    # generate data matrix
    data = pd.DataFrame(index=metadata.index, columns=vardata.index)
    for gsm_name, gsm in gse.gsms.items():
        idx = gsm.table['ID_REF'].values
        values = gsm.table['VALUE'].values
        data.loc[gsm_name, idx] = values

    # grab description of dataset
    description = ' '.join([metadata[col].unique()[0] for col in metadata.filter(regex='protocol*!').columns])

    # remove protocol columns
    meta_cols = metadata.columns[~metadata.columns.str.contains('protocol')]
    metadata = metadata[meta_cols]

    # create dataset object
    ds = DataSet(name='GSE73072',
                 data=data,
                 metadata=metadata,
                 vardata=vardata,
                 description=description,
                 path=data_dir)

    # save to disk
    ds.save()