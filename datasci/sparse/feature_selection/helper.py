import numpy as np
import sys
import pandas as pd
def reduce_feature_set_size(ds, 
                            features_dataframe, 
                            sample_ids,
                            attr:str,
                            classifier_factory_method_handle: str, 
                            scorer, 
                            ranking_method_handle,
                            ranking_method_args: dict,
                            partitioner=None, 
                            test_sample_ids=None,
                            start = 5, 
                            end = 100, 
                            jump = 5, 
                            **kwargs):
    """
    This method takes a features dataframe (outut of a feature selection), ranks them by a ranking method and performs 
    a feature set reduction using grid search method which is defined by start, end and jump parameters. 

    Different training and test data may be used and the results will change accordingly. The following choices are avaible:
    
    if test_sample_ids is None and partitioner is None:
        Only training data is available, so the results will contain score on the Training data

    if test_sample_ids is None and partitioner is not None:
        Model is trained using partitions of sample_ids defined by partitioner, results contain the mean test score obtained during classification on these partitions

    if test_sample_ids is not None and partitioner is None:
        Model is trained on all sample_ids, and then it is then evaluated on test_sample_ids. Results contain evaluation score on test_sample_ids

    if test_sample_ids is not None and partitioner is not None:
        Model is trained using partitions of sample_ids defined by partitioner, and the best model is then evaluated on test_sample_ids. Results
        contain evaluation score on test_sample_ids

    Parameters:
        features_df (pandas.DataFrame): This is a features dataframe that contains result of a feature selection. 
                                        (check datasci.core.dataset.DataSet.feature_select method for details)

        sample_ids (like-like): List of indicators for the samples to use for training. e.g. [1,3], [True, False, True],
            ['human1', 'human3'], etc..., can also be pandas series or numpy array.

        attr (string): Name of metadata attribute to classify on.

        scorer (object): Function which scores the prediction labels on training and test partitions. This function
            should accept two arguments: truth labels and prediction labels. This function should output a score
            between 0 and 1 which can be thought of as an accuracy measure. See
            sklearn.metrics.balanced_accuracy_score for an example.

        ranking_method_handle (method handle) : handle of the feature ranking method

        ranking_method_args (dict): argument dictionary for the feature ranking method

        partitioner (object): Class-instance which partitions samples in batches of training and test split. This
        instance must have the sklearn equivalent of a split method. The split method returns a list of
        train-test partitions; one for each fold in the experiment. See sklearn.model_selection.KFold for
        an example partitioner. (default = None, check the method description above to see how this affects the results)

        test_sample_ids: List of indicators for the samples to use for testing. e.g. [1,3], [True, False, True],
            ['human1', 'human3'], etc..., can also be pandas series or numpy array. (default = None, check the method 
            description above to see how this affects the results)

        start (int): starting point of the grid search. (default: 5)
        
        end (int) :  end point of the grid search. Use -1 to set end as the size of features (default: 100)
        
        jump (int) : gap between each sampled point in the grid (default: 5)

    Return:
    
        a dictionary that contains 2 key-value pairs:
            'optimal_n_results': an array of m x 2 values, with m being the total values sampled from the grid in the search. 
            The first column contains the number of top features (different values sampled from the grid search),  and the second 
            column contains the score.  The array is sorted by score in descending order.
        
            'reduced_feature_ids' : array of reduced features ids(index of features_df). The size of the reduced feature set is the 
            smallest value n, out of the m sampled values, that produces the highest score.

    Example:
            >>> import datasci.core.dataset as DS
            >>> import datasci.sparse.feature_selection.IterativeFeatureRemoval as IFR
            
            >>> x = DS.load_dataset(file_path)
            >>> ifr = IFR.IFR(
                verbosity = 2,
                nfolds = 4,
                repetition = 500,
                cutoff = .6,
                jumpratio = 5,
                max_iters = 100,
                max_features_per_iter_ratio = 2
                )
            >>> result = x.feature_select(ifr,
                        attrname,
                        selector_name='IFR',
                        f_results_handle='results',
                        append_to_meta=False,
                        )
            >>> features_df = results['f_results']

            >>> def model_factory():
                    return svm.LinearSVC(dual=False)

            >>> bsr = sklearn.metrics.balanced_accuracy_score

            >>> ranking_method_args = {'attr': 'frequency'}

            >>> partitioner = KFold(n_splits=5, shuffle=True, random_state=0)

            >>> import datasci.sparse.feature_selection.helper as fhelper

            >>> reduced_feature_results = fhelper.reduce_feature_set_size(x, 
                                    features_df, 
                                    sample_ids_training,
                                    attrname,
                                    model_factory, 
                                    bsr, 
                                    fhelper.rank_features_by_attribute,
                                    ranking_method_args,
                                    test_sample_ids=sample_ids_validation,
                                    start = 5, 
                                    end = 100, 
                                    jump = 1)

            >>> print(reduced_feature_results)
    """
    ranked_features = ranking_method_handle(features_dataframe, ranking_method_args)

    #create subset of features, from "start" to "end" in increments of "jump"
    if end == -1:
        end = ranked_features.shape[0] + 1

    n_attrs = np.arange(start, end+ 1, jump)

    results = np.zeros((n_attrs.shape[0], 2))

    #for each subset of top features
    for i, n in enumerate(n_attrs):

        print("\n\n\n=======================================================")
        print("Using n = ", n, " features")
        print("=======================================================\n\n\n")

        model = classifier_factory_method_handle()   
        classification_result = ds.classify(model,
                    attr,
                    feature_ids=ranked_features[:n],
                    sample_ids=sample_ids,
                    scorer=scorer,
                    partitioner=partitioner,
                    **kwargs
                    )
                
        if test_sample_ids is not None:

            #code duplication. make it a function!
            try:
                data = ds.data[ds.vardata.index[test_sample_ids]]
            except IndexError:
                try:
                    data = ds.data[test_sample_ids]
                except KeyError:
                    data = ds.data[ds.data.columns[test_sample_ids]]

            data = data[ranked_features[:n]].values
            labels = ds.metadata[attr].loc[test_sample_ids]

            predicted_labels = classification_result['classifiers'].values[0].predict(data)
            score = scorer(labels, predicted_labels)

        else:
            if partitioner is not None:
                score = np.mean(classification_result['scores'].loc['Test'].values)
            else:
                score = np.mean(classification_result['scores'].loc['Training'].values)
            
        print('Test score:\t', score)
        results[i, 0] = n
        results[i, 1] = score

    #find the best n, i.e. smallest n that produced largest score
    results = results[results[:,1].argsort()[::-1]]
    max_bsr = np.max(results[:, 1])
    max_bsr_idxs = np.where(results[:, 1] == max_bsr)[0]
    n = int(np.min(results[max_bsr_idxs, 0]))

    reduced_features = features_dataframe.loc[ranked_features[:n]]

    returns = {}
    returns = {'reduced_feature_ids': reduced_features.index.values,
                'optimal_n_results': results}

    return returns


def rank_features_by_attribute(features_df, args):
    """
    This method takes a features dataframe as input and ranks them based on a column/attribute which contains numerical data.  

    Parameters:
        features_df (pandas.DataFrame): This is a features dataframe that contains result of a feature selection. 
                                        (check datasci.core.dataset.DataSet.feature_select method for details)

        args (dict): This dictionary contains variables to determine which attribute to rank feature on and the
                    order of ranking. Check details for various key and values below:
                    'attr' (Mandatory): Attribute/ Column name in the features_df to rank the features on
                    'order': Whether to rank in ascending or descending order. 'asc' for ascending and 'desc' for descending.
                             (defaul: 'desc') 
                    'feature_ids' (list-like): To limit the ranking within certain features. List of indicators for the features to use. 
                            e.g. [True, False, True], ['gene1', 'gene3'], etc..., can also be pandas series or numpy array. 
                            Default: None which corresponds to using all features.
          
    Return:
        array of sorted features (index of features_df) 

    Examples:
            >>> import datasci.core.dataset as DS
            >>> import datasci.sparse.feature_selection.IterativeFeatureRemoval as IFR
            
            >>> x = DS.load_dataset(file_path)
            >>> ifr = IFR.IFR(
                verbosity = 2,
                nfolds = 4,
                repetition = 500,
                cutoff = .6,
                jumpratio = 5,
                max_iters = 100,
                max_features_per_iter_ratio = 2
                )
            >>> result = x.feature_select(ifr,
                        attrname,
                        selector_name='IFR',
                        f_results_handle='results',
                        append_to_meta=False,
                        )
            >>> features_df = results['f_results']
            >>> ranking_method_args = {'attr': 'frequency'}
            >>> ranked_order =  rank_features_by_attribute(features_df, ranking_method_args)
    """
    #create an array whose first column is feature indices and 
    #second column is values of the "attr" 
    if args.get('feature_ids', None) is not None:
        indices = features_df.loc[args['feature_ids']].index.values
    else:
        indices = features_df.index.values
    attr_values = features_df[args['attr']].loc[indices].values.reshape(-1,1)
    #remove features with nan values
    np.isnan(attr_values.astype(float))
    non_nan_idxs = np.invert(np.isnan(attr_values.astype(float))).reshape(-1)

    indices = np.array(indices).reshape(-1, 1)
    feature_array = np.hstack((indices, attr_values))

    feature_array = feature_array[non_nan_idxs, :]

    order=args.get('order', 'desc')
    if order=='desc':
        feature_array = feature_array[feature_array[:,1].argsort()[::-1]]
    elif order=='asc':
        feature_array = feature_array[feature_array[:,1].argsort()]
    else:
        raise ValueError('%s is an incorrect value for rank "order" in args. It should "asc" for ascending or "desc" for descending.'%order)
    
    return feature_array[:, 0]


def rank_features_by_mean_attribute_value(features_df, args):
    """
    CHECK THIS BEFORE COMMITING
    This method takes a features dataframe as input and ranks them based on a column/attribute which contains numerical data.  

    Parameters:
        features_df (pandas.DataFrame): This is a features dataframe that contains result of a feature selection. 
                                        (check datasci.core.dataset.DataSet.feature_select method for details)

        args (dict): This dictionary contains variables to determine which attribute to rank feature on and the
                    order of ranking. Check details for various key and values below:
                    'attr' (Mandatory): Attribute/ Column name in the features_df to rank the features on
                    'order': Whether to rank in ascending or descending order. 'asc' for ascending and 'desc' for descending.
                             (defaul: 'desc')
                    'feature_ids' (list-like): To limit the ranking within certain features. List of indicators for the features to use. 
                            e.g. [True, False, True], ['gene1', 'gene3'], etc..., can also be pandas series or numpy array. 
                            Default: None which corresponds to using all features.
                    'method': which operation to perform. Can be "mean" or "median". (Default: "mean")
          
    Return:
        array of sorted features (index of features_df) 

    Examples:
            >>> import datasci.core.dataset as DS
            >>> import datasci.sparse.feature_selection.IterativeFeatureRemoval as IFR
            
            >>> x = DS.load_dataset(file_path)
            >>> ifr = IFR.IFR(
                verbosity = 2,
                nfolds = 4,
                repetition = 500,
                cutoff = .6,
                jumpratio = 5,
                max_iters = 100,
                max_features_per_iter_ratio = 2
                )
            >>> result = x.feature_select(ifr,
                        attrname,
                        selector_name='IFR',
                        f_results_handle='results',
                        append_to_meta=False,
                        )
            >>> features_df = results['f_results']
            >>> ranking_method_args = {'attr': 'frequency'}
            >>> ranked_order =  rank_features_by_attribute(features_df, ranking_method_args)
    """
    #create an array whose first column is feature indices and 
    #second column is values of the "attr" 
    if args.get('feature_ids', None) is not None:
        indices = features_df.loc[args['feature_ids']].index.values
    else:
        indices = features_df.index.values
    
    #get values
    values = features_df[args['attr']].loc[indices].values
    method = args.get('method', 'mean')

    if method == 'mean':
        #get means
        processed_values = np.array([np.mean(np.array(x)) for x in values ]).reshape(-1, 1)
    else:
        processed_values = np.array([np.median(np.array(x)) for x in values ]).reshape(-1, 1)
    
    #get indices where values is not nan
    idx = np.where(np.isnan(processed_values) == False)[0]
    
    indices = np.array(indices).reshape(-1, 1)
    feature_array = np.hstack((indices, processed_values))
    feature_array = feature_array[idx, :]

    order=args.get('order', 'desc')
    if order=='desc':
        feature_array = feature_array[feature_array[:,1].argsort()[::-1]]
    elif order=='asc':
        feature_array = feature_array[feature_array[:,1].argsort()]
    else:
        raise ValueError('%s is an incorrect value for rank "order" in args. It should "asc" for ascending or "desc" for descending.'%order)
    
    return feature_array[:, 0]


def rank_features_within_attribute_class(features_df, 
                                    feature_class_attribute, 
                                    new_feature_attribute_name,
                                    x, 
                                    partitioner, 
                                    sample_ids,
                                    scorer,
                                    classification_attr,
                                    classifier_factory_method,
                                    f_weights_handle,
                                    feature_ids = None,
                                    **kwargs):
    features_df[new_feature_attribute_name] = 0

    #get ranked features
    # args = {'attr': feature_class_attribute}
    # if feature_ids is not None:
    #     args['feature_ids'] = feature_ids
    # ranked_features = rank_features_by_attribute(features_df, args)

    if feature_ids is None:
        feature_ids = features_df.index
    #get unique values (classes) for the attributes
    unique_attr_values = np.unique(features_df[feature_class_attribute].loc[feature_ids].values)
    unique_attr_values = np.sort(unique_attr_values)[::-1]
    for val in unique_attr_values:
        features = features_df[feature_class_attribute] == val
        if features.sum() != 1:    

            #train model on the features for the current class
            classififer = classifier_factory_method()
            results = x.classify(classififer,
                        classification_attr,
                        feature_ids=features,
                        sample_ids=sample_ids,
                        partitioner=partitioner,
                        scorer=scorer,
                        f_weights_handle = f_weights_handle)

            #extract mean of absolute weights for features
            weights_keys = results['f_weights'].keys()
            filtered_keys = [k for k in weights_keys if 'weights' in k]            
            weights = results['f_weights'][filtered_keys].values
            mean_of_abs_weights = np.mean(np.abs(weights), axis=1)

            #normalize features between 0 and 1
            
            b = .999
            a = .001
            r_max = np.max(mean_of_abs_weights)
            r_min = np.min(mean_of_abs_weights)
            mean_of_abs_weights = (b-a) * (mean_of_abs_weights - r_min) / (r_max - r_min) + a
            mean_of_abs_weights = val + mean_of_abs_weights
        else:
            mean_of_abs_weights = val
            
        features_df.loc[features, new_feature_attribute_name] = mean_of_abs_weights


        

