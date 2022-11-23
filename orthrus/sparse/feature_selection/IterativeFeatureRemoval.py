# from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import os
from orthrus.solvers.linear import LPPrimalDualPy
import copy 
from sklearn.metrics import balanced_accuracy_score
from orthrus.core.helper import batch_jobs_, reconstruct_logger_from_details, extract_reconstruction_details_from_logger
import ray
import copy
import logging 

class IFR:

    """
    The Iterative Feature Removal algorithms is used to extract ordered disctiminatory feature sets for classification problems. The feature are ranked
    is 'frequency', which marks marks how important a feature is. Internally, IFR randomly partitions the data many times, and computes 
    one feature set for each data partition. During execution, the data is partitioned, using the :py:attr:`partitioner` object, into training and validation sets 
    :py:attr:`repetition` number of times. Then for each partition, one feature set is extracted. So, a total of repetition * number_of_partitions feature sets 
    are extracted. These feature sets, which are independent of each other are combined into one final feature set, and the 
    reoccurence of any features are noted as 'frequency'. The feature set extraction on each partition is independent of each other, and so are batched 
    and run in parallel using the `ray` package. One feature extraction on a data partition is a ray worker. See below to check how to specify 
    resource requirements for each worker.
    
    For each individual feature set on a data partition, the algorithm can halt because of the following conditions:

        1. Score on validation partition is below cutoff 
        2. Jump does not occur in sorted absolute weights
        3. Jump occurs, but the weight at the jump is too small ( < 10e-6)
        4. Number of features selected for the current iteration is greater than number of samples in training partition - 1. This condition prevents overfitting.

    When one of these conditions happen, further feature extraction on the current fold is stopped.

    Parameters:

        classifier (object): Classifier to run the classification experiment with; must have the sklearn equivalent
                of a ``fit`` and ``predict`` method. Default classifier is orthrus.sparse.classifiers.svm .SSVMClassifier, it will
                be a CPU based classifier if ``num_gpus_per_task`` is 0, otherwise it will be a GPU classifier.
        
        weights_handle (str) : Name of ``classifier`` attribute containing feature weights. Default is 'weights_'.

        partitioner (object): Class-instance which partitions samples in batches of training and test
                split. This instance must have the sklearn equivalent of a split method and must contain 'shuffle' and
                'random_state' attributes. The split method returns a
                list of train-test partitions; one for each fold in the experiment. See sklearn.model_selection.KFold
                for an example partitioner. Default is sklearn.model_selection.StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

        scorer (object): Function which scores the prediction labels on training and test partitions. This function
            should accept two arguments: truth labels and prediction labels. This function should output a score
            between 0 and 1 which can be thought of as an accuracy measure. See
            sklearn.metrics.balanced_accuracy_score for an example.

        repetition (int): Determines the number of times to partition the dataset. (default: 10)
        
        validation_score_cutoff (float): Threshold for the validation score to halt the process. (default: 0.75)

        jumpratio (float): The relative drop in the magnitude of coefficients in weight vector to identify numerically zero weights (default: 100)

        verbosity (int) : Determines verbosity of print statments; 0 for no output; 2 for full output. (default: 0)
        
        verbose_frequency (int) : this parameter controls the frequency of progress outputs for the ray workers to console; an output is 
            printed to console after every verbose_frequency number of processes complete execution. (default: 10)

        num_cpus_per_task (float) : Number of CPUs each worker needs. This can be a fraction, check 
            `ray specifying required resources <https://docs.ray.io/en/master/walkthrough.html#specifying-required-resources>`_ for more details. (default: 1.)

        num_gpus_per_task (float) : Number of GPUs each worker needs. This can be fraction, check 
            `ray specifying required resources <https://docs.ray.io/en/master/walkthrough.html#specifying-required-resources>`_ for more details. (default: 0.)
                
        local_mode (bool) : A flag to set whether to initialize Ray in local mode. (default: False)

        null_model_repetitions : The null model for IFR runs a feature selection on the given problem, but with randomly shuffled labels. The aim is to identify
            and remove the frequency classes which occur for the random labels. The null model can be run multiple times and this parameter controls how many 
            times to run. If null model is not required, set the value to 0. (default: 0)

    Attributes:
        diagnostic_information_ (dict): Holds execution the following information for each interation of each partition.
            'train_scores' (list) : Each element is a list of training scores for the feature selection on one data partatition, 
                                    the number of elements in this inner list is the number of iterations IFR ran for, for this particular data partition. 

            'validation_scores' (list): Each element is a list of test scores for the feature selection on one data partatition, 
                                    the number of elements in this inner list is the number of iterations IFR ran for, for this particular data partition.  

            'sorted_abs_weights (list)': Each element is a list of sorted absolute weights for the classifier for the feature selection on one data partatition, 
                                    the number of elements in this inner list is the number of iterations IFR ran for, for this particular data partition.   

            'weight_ratios' (list)': Each element is a list of weight ratios for the classifier for the feature selection on one data partatition, 
                                    the number of elements in this inner list is the number of iterations IFR ran for, for this particular data partition.   

            'features' (list)': Each element is a list of selected feature ids for the feature selection on one data partatition, 
                                    the number of elements in this inner list is the number of iterations IFR ran for, for this particular data partition.   

            'true_feature_count' (list): Each element is a list of true number of features that IFR determined were to be selected for the feature selection on one data partatition, 
                                    the number of elements in this inner list is the number of iterations IFR ran for, for this particular data partition. 

            'stopping_conditions' (list): Each element contains the reason for why the IFR stopped for the feature selection on one data partatition. These are one of the following reasons:
                1. Exception in model fitting
                2. Validation score cutoff: Score on validation partition is below cutoff 
                3. Jump not found: Jump does not occur in the array of sorted absolute weights
                4. Small weight at jump: Jump occurs but the weight at the jump is too small ( < 10e-6)
                5. More features than cutoff: Number of features selected for the current iteration is greater than number of samples in training partition - 1. This condition prevents overfitting.

    Examples:
            >>> import orthrus.core.dataset as DS
            >>> from orthrus.sparse.feature_selection.IterativeFeatureRemoval import IFR
            >>> from orthrus.solvers.linear import LPPrimalDualPy
            >>> from orthrus.sparse.classifiers.svm import SSVMClassifier
            >>> x = DS.load_dataset('path/to/gse_730732.h5')
            >>> model = SSVMClassifier(C=1, solver=LPPrimalDualPy, use_cuda=True)
            >>> from sklearn.model_selection import LeaveOneOut                    
            >>> # define leave one out
            >>> loo = Partition(process=LeaveOneOut(),
            ...     process_name='LOO',
            ...     verbosity=2)
            >>> ifr = IFR(
            ...     model,
            ...     partitioner = loo,
            ...     repetition = 50,
            ...     validation_score_cutoff = .80,
            ...     jumpratio = 5,
            ...     verbosity = 2,
            ...     num_cpus_per_task=1,
            ...     num_gpus_per_task=.2,
            ...     null_model_repetitions=10,
            ...     local_mode=False
            ...     )

            >>> #see feature select method for details
            >>> result = x.feature_select(ifr,
            ...     attrname,
            ...     selector_name='IFR',
            ...     f_results_handle='results_',
            ...     append_to_meta=False,
            ...     )
    
    See :py:meth:`IFR.fit` to understand the output of `IFR`.
    """
    def __init__(self,
                classifier = None,
                weights_handle: str ='weights_',
                partitioner = None,
                scorer = balanced_accuracy_score,
                repetition: int=10,
                validation_score_cutoff: float=0.75,
                jumpratio: float=100,
                verbosity: int=0,
                verbose_frequency: int=10,
                num_cpus_per_task: float=1.,
                num_gpus_per_task: float=0.,
                num_cpus_for_job: float=-1,
                num_gpus_for_job: float=-1,
                local_mode=False,
                null_model_repetitions = 0):

        self.num_cpus_per_task = num_cpus_per_task
        self.num_gpus_per_task = num_gpus_per_task

        self.num_cpus_for_job = num_cpus_for_job
        self.num_gpus_for_job = num_gpus_for_job

        if classifier == None:
            from orthrus.sparse.classifiers.svm import SSVMClassifier
            if self.num_gpus_per_task == 0:
                use_cuda = False
            elif self.num_gpus_per_task > 0:
                use_cuda  = True
            classifier = SSVMClassifier(C=1, solver=LPPrimalDualPy, use_cuda=use_cuda)
        
        if partitioner is None:
            from sklearn.model_selection import StratifiedKFold
            self.partitioner = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        else:
            partitioner.shuffle = True
            self.partitioner =  partitioner 


        self.classifier = classifier
        self.scorer = scorer
        self.weights_handle = weights_handle
        self.repetition = repetition   # Number of time the data is randomly partitioned.
        self.validation_score_cutoff = validation_score_cutoff    # validation score threshold
        self.jumpratio = jumpratio # Relative drop needed to detect numerically zero weights in SSVM.
        self.verbosity = verbosity    # Verbosity of print statements; make positive to see detail.
        self.verbose_frequency = verbose_frequency
        self.local_mode = local_mode
        self.null_model_repetitions = null_model_repetitions
        self.feature_frequency_cutoff_ = None

        self.diagnostic_information_ = {}
        self._diagnostic_information_keys = ['train_scores', 'validation_scores', 'sorted_abs_weights', 'weight_ratios',
                                            'features', 'true_feature_count', 'num_training_samples']
        
        self.diagnostic_information_['stopping_conditions'] = []

        self.logger_info =  extract_reconstruction_details_from_logger(logging.getLogger())

        super(IFR, self).__init__()
    #

    def _initialize_diagnostic_dictionary(self, diag_dict):
        for key in self._diagnostic_information_keys:
            diag_dict[key] = []
       

    def _add_diagnostic_info_for_current_iteration(self, diag_dict, train_score, validation_score,
        sorted_abs_weights, weight_ratios, features, true_feature_count, n_training_samples):

        diag_dict.get('train_scores', []).append(train_score)
        diag_dict.get('validation_scores', []).append(validation_score)
        diag_dict.get('sorted_abs_weights', []).append(sorted_abs_weights)
        diag_dict.get('weight_ratios', []).append(weight_ratios)
        diag_dict.get('features', []).append(features)
        diag_dict.get('true_feature_count', []).append(true_feature_count)
        diag_dict.get('num_training_samples', []).append(n_training_samples)


    def _sanity_check_diagnostics(self, diag_dict, n_iters):
        arr = np.zeros(len(self._diagnostic_information_keys))
        for i, key in enumerate(self._diagnostic_information_keys):
            arr[i] = len(diag_dict[key])

        assert np.unique(arr).shape[0] == 1, 'Lenghts of lists for the diagnostic information do not match. They should be of same size' 
        if n_iters > 1:
            assert np.unique(arr)[0] == n_iters, 'diagnostic dictionary does not contain all the information'


    def _add_diagnostic_info_for_data_partition(self, diag_dict, n_data_partition, stopping_condition):
        for key in self._diagnostic_information_keys:
            self.diagnostic_information_[key].append(diag_dict[key])
        
        self.diagnostic_information_['stopping_conditions'].append(stopping_condition)


    def _reformat_diagnostic_information(self):

        self.diagnostic_information_['train_scores'] = pd.DataFrame(self.diagnostic_information_['train_scores']).transpose()
        self.diagnostic_information_['validation_scores'] = pd.DataFrame(self.diagnostic_information_['validation_scores']).transpose()
        self.diagnostic_information_['true_feature_count'] = pd.DataFrame(self.diagnostic_information_['true_feature_count']).transpose()
        
        # In old saved IFR objects diagnostic information dictionary contains 'exit_reasons' key instead of 'stopping_conditions'stopping_conditions
        try:
            stopping_conditions = self.diagnostic_information_['stopping_conditions']
        except AttributeError:
            stopping_conditions = self.diagnostic_information_['exit_reasons']

        stopping_conditions,  counts = np.unique(np.array(stopping_conditions), 
                                                return_counts=True)

        self.diagnostic_information_['stopping_conditions'] = pd.DataFrame({'Stopping Conditions': stopping_conditions, 'Frequency': counts})
        self.diagnostic_information_['stopping_conditions'].set_index('Stopping Conditions', inplace=True)

        self.diagnostic_information_['number_of_iters_per_run'] = pd.Series([
                self.diagnostic_information_['train_scores'][i].count() 
                    for i in range(len(self.diagnostic_information_['train_scores'].columns))
                ])

        self.diagnostic_information_['weight_ratios'] = pd.DataFrame(self.diagnostic_information_['weight_ratios']).transpose()
        self.diagnostic_information_['sorted_abs_weights'] = pd.DataFrame(self.diagnostic_information_['sorted_abs_weights']).transpose()
        self.diagnostic_information_['num_training_samples'] = pd.DataFrame(self.diagnostic_information_['num_training_samples']).transpose()


    def _initialize_results(self, n_attributes):
        '''
        Initializes self.results_ attribute
        '''
        self.results_ = pd.DataFrame(index=np.arange(n_attributes))
        self.results_['frequency'] = 0
        self.results_['weights'] = np.empty((len(self.results_), 0)).tolist() 
        self.results_['selection_iteration'] = np.empty((len(self.results_), 0)).tolist() 


    def _update_frequency_in_results(self, features):
        '''
        Increments the values of features by 1, for passed features, in frequency column of self.results_
        '''
        self.results_.loc[features, 'frequency'] =  self.results_.loc[features, 'frequency'] + 1


    def _update_selection_iteration_in_results(self, list_of_features):
        '''
        Appends the iteration number, for passed features, in selection_iteration column of self.results_
        '''
        for iteration, features in enumerate(list_of_features):
            for feature in features:
                iter_list = self.results_.loc[feature, 'selection_iteration']
                iter_list.append(iteration)


    def _update_weights_in_results(self, features, weights):
        '''
        Appends the weight for passed features in the weights column in self.results_
        ''' 
        #iterate over features
        for feature, weight in zip(features, weights):
            #append the weight to the list of weights for the current feature
            weights = self.results_.loc[feature, 'weights']
            weights.append(weight)


    def fit(self, X, y, groups=None, **kwargs):
        logger = logging.getLogger( f'{__name__}_pid-{os.getpid()}')
        '''
        Args:
        X (ndarray of shape (m, n))): array of data, with m the number of observations in R^n.
        y (ndarray of shape (m))): vector of labels for the data

        Return:
        (pandas.DataFrame) : The dataframe contains the results of IFR. It is indexed by feature_ids and each column 
        contains different information for each feature as described below:

            * frequency \: How many times the feature is extracted
            * weights \: Contains a list of weights, from the weight vectors during training on different partitions. 
              Each value corresponds to the weight for the feature over different extractions. The length of the weights 
              is equal to the frequency.
            * selection_iteration \: Contains a list of indices of the iteration when the feature was extracted over 
              different data partitions. The length of the list is equal to the frequency.
        '''
        import numpy as np
        from pandas import DataFrame, Series

        if type(y) == Series:
            y  = y.values
        
        if type(X) == DataFrame:
            X  = X.values
        
        if self.verbosity>0:
            logger.info('IFR parameters:')
            logger.info(f'classifier {type(self.classifier)}')
            logger.info(f'scorer {type(self.scorer)}')
            logger.info(f'partitioner {type(self.partitioner)}')
            logger.info(f'repetition {self.repetition}')
            logger.info(f'jumpratio {self.jumpratio}')
            logger.info(f'validation score cutoff {self.validation_score_cutoff}')
            logger.info(f'verbosity{self.verbosity}')

        #

        if self.null_model_repetitions > 0:
            self.determine_frequency_cutoff(X, y, groups)

        self._initialize_results(X.shape[1])
        self._initialize_diagnostic_dictionary(self.diagnostic_information_)

        self.compute_features(X, y, groups, self.partitioner, save_diagnostic=True)

        if self.verbosity>0:
            logger.info("=====================================================")
            logger.info(f"Finishing Execution. {(self.results_['frequency'] > 0).sum()} features out of a total of {X.shape[1]} features were selected.")
            logger.info("=====================================================")

        self._reformat_diagnostic_information()
        return self

    @ray.remote
    def select_features_for_data_partition(self, train_X, validation_X, train_y, validation_y, logger_info, save_diagnostic=True):
        
        # This method runs in a new process, and so, this process's root logger does not have any file handles attached to it.
        # Therefore, we need to create the logger for this method using logger_info which will add the file handlers, if there are any, to this logger.
        logger = reconstruct_logger_from_details(logger_info, __name__, add_pid='always')

        _, n = train_X.shape
        list_of_features_for_curr_fold = np.array([], dtype=np.int64)
        list_of_weights_for_curr_fold = np.array([], dtype=np.int64)  
        list_of_selection_iterations_for_current_fold = []

        selected = np.array([], dtype=np.int64)
        # Mask array which tracks features which haven't been removed.
        active_mask = np.ones(n, dtype=bool)

        if save_diagnostic:
            #create an empty dictionary to store diagnostic info for the
            #current data partition, this dictionary has info about each iteration
            #on the current data partition
            diagnostic_info_dictionary = {}
            self._initialize_diagnostic_dictionary(diagnostic_info_dictionary)
        i = 0
        while(True):
            if self.verbosity > 1:
                logger.info("=====================================================")
                logger.info(f"beginning of inner loop iteration {i+1}")
                logger.info(f"Number of features selected for this fold: {len(list_of_features_for_curr_fold)} of {n}")
                logger.info("Checking score of complementary problem... ")
            #

            #create a copy of the classifier
            model = copy.deepcopy(self.classifier)

            tr_d = np.array( train_X[:,active_mask] )
            te_d = np.array( validation_X[:,active_mask] )
            try:
                model.fit(tr_d, train_y)
            except Exception as e:
                if self.verbosity>0:
                    logger.error("Exception occurred during fitting the model in IFR:")
                    logger.error(e, exc_info=True)
                    logger.error("Breaking the execution for the current data fold")
                    #save the diagnostic information
                if save_diagnostic:    
                    self._add_diagnostic_info_for_current_iteration(diagnostic_info_dictionary,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        tr_d.shape[0])
                    stopping_condition = "Exception in model fitting"
                break
            
            weight = eval("model" + "." + self.weights_handle)

            #calculate score for training data
            pred_train = model.predict(tr_d)
            score_train = self.scorer(train_y, pred_train)

            ##########
            #
            # Detect where the coefficients in the weight vector are
            # numerically zero, based on the (absolute value) ratio of
            # successive coefficients.
            #

            # Look at absolute values and sort largest to smallest.
            abs_weights = (np.abs(weight)).flatten()
            order = np.argsort(-abs_weights)
            sorted_abs_weights = abs_weights[order]

            # Detect jumps in the coefficient values using a ratio parameter.
            weight_ratios = sorted_abs_weights[:-1] / (sorted_abs_weights[1:] + np.finfo(float).eps)
            jumpidxs = np.where(weight_ratios > self.jumpratio)[0]

            if self.verbosity>1:
                logger.info("Training Score %.3f. "%score_train)

            #calculate score for validation data
            pred_validation = model.predict(te_d)
            score_validation = self.scorer(validation_y, pred_validation)

            if self.verbosity>1:
                logger.info("Validation Score %.3f. "%score_validation)

            #Check if score is above cutoff
            if (score_validation < self.validation_score_cutoff):
                if self.verbosity>1:
                    logger.info("Validation score below cutoff, exiting inner loop.")

                if save_diagnostic:
                    #save the diagnostic information for this iteration
                    #in this case we only have train and validation score
                    self._add_diagnostic_info_for_current_iteration(diagnostic_info_dictionary,
                        score_train,
                        score_validation,
                        sorted_abs_weights,
                        weight_ratios,
                        None,
                        None,
                        tr_d.shape[0])

                    #break out of current loop if score is below cutoff
                    stopping_condition = "Validation score cutoff"
                break




            #check if sufficient jump was found
            if len(jumpidxs)==0:
                #jump never happened.
                #save the diagnostic information for this iteration
                #we still do not have the selected feature count and features
                if save_diagnostic:
                    self._add_diagnostic_info_for_current_iteration(diagnostic_info_dictionary,
                        score_train,
                        score_validation,
                        sorted_abs_weights,
                        weight_ratios,
                        None,
                        None,
                        tr_d.shape[0])
                    stopping_condition = "Jump not found"
            
            
                #break out of the loop
                if self.verbosity>1:
                    logger.info('There was no jump of sufficient size between ratios of successive coefficients in the weight vector.')
                    logger.info("Discarding iteration..")
                break

            else:
                count = jumpidxs[0]

            #check if the weight at the jump is greater than cutoff
            if sorted_abs_weights[count] < 1e-6:
                if save_diagnostic:
                    self._add_diagnostic_info_for_current_iteration(diagnostic_info_dictionary,
                        score_train,
                        score_validation,
                        sorted_abs_weights,
                        weight_ratios,
                        None,
                        None,
                        tr_d.shape[0])
                    stopping_condition = "Small weight at jump"
                if self.verbosity>1:
                    logger.info(f'Weight at the jump({sorted_abs_weights[count]})  smaller than weight cutoff(1e-6).')
                    logger.info("Discarding iteration..")
                break

            count += 1
            #check if the number of selected features is greater than the cap
            if count >= train_X.shape[0] - 1:
                if save_diagnostic:
                    self._add_diagnostic_info_for_current_iteration(diagnostic_info_dictionary,
                        score_train,
                        score_validation,
                        sorted_abs_weights,
                        weight_ratios,
                        None,
                        None,
                        tr_d.shape[0])
                    stopping_condition = "More features than cutoff"
                if self.verbosity>1:
                    logger.info(f'More features selected ({count}) than the cutoff ({train_X.shape[0]})')
                    logger.info("Discarding iteration..")
                
                break

            
            #select features: order is list of sorted features
            selected = order[:count]

            if self.verbosity>1:
                logger.info("Selected features on this iteration:")
                logger.info(str(selected))
            #

            # Selected indices are relative to the current active set.
            # Get the mapping back to the original indices.
            active_idxs = np.where(active_mask)[0]

            active_mask[active_idxs[selected]] = 0

            #append the selected features to the list_of_features_for_curr_fold
            list_of_features_for_curr_fold = np.concatenate([list_of_features_for_curr_fold ,  active_idxs[selected]])
            list_of_weights_for_curr_fold = np.concatenate([list_of_weights_for_curr_fold ,  weight.flatten()[order][:count]])
            
            if save_diagnostic:
                #save the diagnostic information for this iteration
                #here we have all the information we need
                self._add_diagnostic_info_for_current_iteration(diagnostic_info_dictionary,
                    score_train,
                    score_validation,
                    sorted_abs_weights,
                    weight_ratios,
                    active_idxs[selected],
                    count,
                    tr_d.shape[0])

            if self.verbosity>1:
                logger.info(f'Removing {len(selected)} features from training and validation matrices.')

            #append the selection iterations for the features selected in the current iteration
            list_of_selection_iterations_for_current_fold.append(active_idxs[selected])

            i+=1
        
        results = {}
        results['list_of_features'] = list_of_features_for_curr_fold
        results['list_of_weights'] = list_of_weights_for_curr_fold
        results['list_of_selection_iteration'] = list_of_selection_iterations_for_current_fold
        if save_diagnostic:
            results['diagnostic_info_dictionary'] = diagnostic_info_dictionary
            results['stopping_condition'] = stopping_condition
        results['n_iters'] = i+1
        return results 


    def compute_features(self, X, y, groups, partitioner, save_diagnostic=True):
        logger = logging.getLogger( f'{__name__}_pid-{os.getpid()}')
        logger.info('Starting feature selection.')
        n_data_partition = 0
        list_of_arguments = []
        for n_rep in range(self.repetition):
            
                partitions = partitioner.split(X, y, groups)
                try:
                    partitioner.random_state += 1
                except:
                    pass
                for i, partition in enumerate(partitions):

                    n_data_partition +=1
                    train_idx, validation_idx = partition

                    train_X = X[train_idx, :]
                    train_y = y[train_idx]

                    validation_X = X[validation_idx, :]
                    validation_y = y[validation_idx]
                    
                    arguments = [self,
                                train_X,
                                validation_X,
                                train_y,
                                validation_y,
                                self.logger_info,
                                save_diagnostic]

                    list_of_arguments.append(arguments)
 
        all_results = batch_jobs_(self.select_features_for_data_partition, 
                                list_of_arguments, 
                                verbose_frequency=self.verbose_frequency,
                                num_cpus_per_task=self.num_cpus_per_task, 
                                num_gpus_per_task=self.num_gpus_per_task, 
                                num_gpus_for_job=self.num_gpus_for_job,
                                num_cpus_for_job=self.num_cpus_for_job,
                                local_mode=self.local_mode)

        for results in all_results:
            # update the feature set dictionary based on the features collected for current fold
            list_of_features_for_curr_fold = results['list_of_features']
            self._update_frequency_in_results(list_of_features_for_curr_fold)
            
            list_of_weights_for_curr_fold = results['list_of_weights']
            self._update_weights_in_results(list_of_features_for_curr_fold, list_of_weights_for_curr_fold) 

            list_of_selection_iterations_for_current_fold = results['list_of_selection_iteration']
            self._update_selection_iteration_in_results(list_of_selection_iterations_for_current_fold)               
            #`

            if save_diagnostic:
                n_iters = results['n_iters']
                diagnostic_info_dictionary = results['diagnostic_info_dictionary']
                stopping_condition = results['stopping_condition']
                                
                self._sanity_check_diagnostics(diagnostic_info_dictionary, n_iters)
                #save the diagnostic information for this data partition
                self._add_diagnostic_info_for_data_partition(diagnostic_info_dictionary, n_data_partition, stopping_condition)


    def determine_frequency_cutoff(self, X, y, groups):
        logger = logging.getLogger( f'{__name__}_pid-{os.getpid()}')
        if self.verbosity>0:
            logger.info("=====================================================")
            logger.info("Determining optimal frequency cutoff value. Running IFR on random variables.")
            logger.info("=====================================================")

        self._initialize_results(X.shape[1])
        y_copy = copy.deepcopy(y)
        if groups is not None:
            groups_copy = copy.deepcopy(groups)
        else:
            groups_copy = None
        partitioner = copy.deepcopy(self.partitioner)

        for j in range(self.null_model_repetitions):
            
            #randomize labels and groups
            if groups is not None:
                #shuffle groups
                np.random.shuffle(groups_copy)

                # generate new labels for each group
                unique_y = np.unique(y)
                unique_groups = np.unique(groups)

                # get probability of class at index 1
                p = np.where(y == unique_y[1])[0].shape[0] / y.shape[0]
                new_label_idxs = np.random.binomial(1, p, size=unique_groups.shape[0])
                new_labels = [unique_y[x] for x in new_label_idxs]

                #assign new label for each group
                for group, label in zip(unique_groups, new_labels):
                    #for all samples of the current group, randomly assign a new label
                    new_idx_for_group = np.where(groups_copy == group)[0]

                    old_idx_for_group = np.where(groups == group)[0]
                    old_label = y[old_idx_for_group][0]
                    y_copy[new_idx_for_group] = label
                    # print(old_label, label, y_copy[new_idx_for_group])

            else:
                np.random.shuffle(y_copy)

            if self.verbosity>0:
                logger.info("=====================================================")
                logger.info(f"Null model run #: {j}")
                logger.info("=====================================================")
            #

            # start processing
            self.compute_features(X, y_copy, groups_copy, partitioner, save_diagnostic=False)
            
        self.frequency_cutoff_results_ = copy.deepcopy(self.results_)

        self.frequency_cutoff_results_['frequency'] = self.frequency_cutoff_results_['frequency'] / self.null_model_repetitions
        
        self.feature_frequency_cutoff_ = self.frequency_cutoff_results_['frequency'].max()

        if self.verbosity>0:
            logger.info("=====================================================")
            logger.info(f"Optimal value for frequency cutoff determined to be {self.feature_frequency_cutoff_}. Now running the feature selection for the actual problem.")
            logger.info("=====================================================")
        # ray.shutdown()

        return self
 

    def transform(self, features, **kwargs):
        cutoff = self.feature_frequency_cutoff_ if self.null_model_repetitions > 0 else 0
        return self.results_['frequency'] > cutoff

    def launch_dashboard(self, kwargs=None):
        import plotly.io as pio
        import dash
        from dash import html
        from dash import dcc
        from dash.dependencies import Input, Output
        import plotly.graph_objects as go
        import plotly.express as px
        import dash_bootstrap_components as dbc
        logger = logging.getLogger(__name__)

        #for old saved ifr objects, diagnostic information needs to be reformatted
        if type(self.diagnostic_information_['train_scores']) == list:
            self._reformat_diagnostic_information()

        self.plotly_colors = {
                'background': '#111111',
                'text': '#7FDBFF',
                'train_line': 'rgba(101,110,242, 1)',
                'train_fill': 'rgba(101,110,242, 0.3)',
                'validation_line': 'rgba(221,96,70, 1)',
                'validation_fill': 'rgba(221,96,70, 0.3)',    
                'cutoff_line': 'rgba(246,205,104, 1)',
        }

        pio.templates.default = "plotly_dark"
        # bootstrap theme
        # https://bootswatch.com/lux/
        external_stylesheets = [dbc.themes.CYBORG]

        app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

        n_ifr_runs = len(self.diagnostic_information_['train_scores'].columns)

        app.layout = html.Div([
                    dbc.Container([
                        dbc.Row([
                            dbc.Col(html.H1("ITERATIVE FEATURE REMOVAL"), className="text-center")
                        ]),
                        
                        dbc.Row([
                            dbc.Col(html.H5(children='Diagnostic Dashboard'), className="text-center mb-2")
                        ]),


                        dbc.Row([
                            dbc.Col(dcc.Tabs(id="tab-navbar", 
                                            value='tab-1', 
                                            children=[
                                                dcc.Tab(label='Stopping Conditions', value='tab-1'),
                                                dcc.Tab(label='Scores', value='tab-2'),
                                                dcc.Tab(label='Features', value='tab-3'),
                                                dcc.Tab(label='Weight Vectors', value='tab-4')], 
                                            colors={
                                                "border": "#111111",
                                                "primary": self.plotly_colors['validation_line'],
                                                "background": "#181818"}), 
                                    className="text-center mb-4")
                        ]),

                        #Empty container where plots are loaded using _update_main_container_layout callback
                        html.Div(id = 'main_container_div'),
                    ])
            ])


        @app.callback(Output('main_container_div', 'children'),
                    Input('tab-navbar', 'value'))
        def _update_main_container_layout(tab_value):
    
            if tab_value == 'tab-1':

                return html.Div([
                ######################################################
                ####### Exit Reasons
                ######################################################    
                dbc.Row([
                    dbc.Col(dbc.Card(html.H3(children='Histogram of Stopping Conditions', className="text-center"),
                                     body=True), 
                            className="mb-4")
                ]),


                dbc.Row([
                    dbc.Col(dcc.Graph(id='stopping_conditions',
                                      figure= px.bar(self.diagnostic_information_['stopping_conditions'], 
                                                    x="Stopping Conditions", 
                                                    y="Frequency")))
                ]),


                ######################################################
                ####### Distribution of Number of Iterations
                ######################################################
                dbc.Row([
                    dbc.Col(dbc.Card(html.H3(children='Distribution of Number of Iterations', className="text-center"),
                                     body=True), 
                            className="mb-4", 
                            style={'padding-top':'100px'})
                ]),
                dbc.Row([
                    dbc.Col(html.Label('Select number of bins:'), width=2),
                    dbc.Col(dcc.Slider(id='number_of_iters_bins',
                                        min=1,
                                        max=30,
                                        step=1,
                                        value=0,
                                        marks={str(i): str(i) for i in range(0, 30, 1)}))
                ], style={'display': 'none'}),

                dbc.Row([
                    dbc.Col(dcc.Graph(id='number_of_iters_hist'))
                ]),
                html.Hr(),
                ])
                
            elif tab_value == 'tab-2':
                return html.Div([
                ######################################################
                ####### Classification Scores
                ######################################################
                dbc.Row([
                    dbc.Col(dbc.Card(html.H3(children='Classification Scores',
                                            className="text-center"), 
                                    body=True), 
                            className="mb-4")
                ]),
                dbc.Row([
                    dbc.Col([
                            html.H6('Type of Plot', className="form-label"),
                            dcc.RadioItems(id='score_radio',
                                            options=[
                                                {'label': 'Mean Curve', 'value': 'mean'},
                                                {'label': 'Individual Curves', 'value': 'individual'},
                                            ],
                                            value='mean',
                                            labelStyle={'display': 'block'},
                                            inputStyle={"margin-right": "5px", "margin-left": "15px" })
                        ], 
                        width=2),

                    dbc.Col(
                        html.Div([
                            html.H6('Select number of individual IFR runs:', className="form-label"),
                            dcc.Slider(
                                id='score_slider',
                                min=1,
                                max=n_ifr_runs,
                                step=1,
                                value=n_ifr_runs,
                                marks={str(i): str(i) for i in range(1, 
                                                                    n_ifr_runs, 
                                                                    int(n_ifr_runs / 10))},
                                tooltip={"placement": "bottom"},
                                updatemode='drag')
                            ]), 
                        width=8),

                    dbc.Col(html.Button('Randomize', 
                                        className='form-label btn btn-primary', 
                                        style={"float": 'right', 
                                                'background': self.plotly_colors['train_line'],  
                                                'border-color': self.plotly_colors['train_line']},  
                                        id='score_trials_randomize_button'), 
                        width=2)
                ]),

                dbc.Row([
                    dbc.Col(dcc.Graph(id='classification_scores'))
                    ], 
                    style={'padding-top':'15px'}),


                ######################################################
                ####### Distribution of Validation Scores
                ######################################################
            
                dbc.Row([
                    dbc.Col(dbc.Card(html.H3(children='Distribution of Validation Scores',
                                            className="text-center"), 
                                    body=True), 
                            className="mb-4", 
                            style={'padding-top':'100px'})
                ]),
                dbc.Row([
                    dbc.Col(html.Label('Select number of bins:'), width=2),
                    dbc.Col(dcc.Slider(
                                id='validation_score_hist_bins',
                                min=1,
                                max=30,
                                step=1,
                                value=0,
                                marks={str(i): str(i) for i in range(0, 30, 1)}))
                    ], 
                    style={'display': 'none'}),

                dbc.Row([
                    dbc.Col(dcc.Graph(id='validation_score_hist'))
                    ]),

                html.Hr(),
            ])
            elif tab_value == 'tab-3':
                return html.Div([
                ######################################################
                ####### Number of features per Iteration
                ######################################################

                dbc.Row([
                    dbc.Col(dbc.Card(html.H3(children='Number of Features per Iteration',
                                            className="text-center"), 
                                    body=True), 
                            className="mb-4")
                ]),

                dbc.Row([
                    dbc.Col([
                            html.H6('Type of Plot', className="form-label"),
                            dcc.RadioItems(id='features_per_iter_radio',
                                            options=[
                                                {'label': 'Mean Curve', 'value': 'mean'},
                                                {'label': 'Individual Curves', 'value': 'individual'},
                                            ],
                                            value='mean',
                                            labelStyle={'display': 'block'},
                                            inputStyle={"margin-right": "5px", 
                                                        "margin-left": "15px" })
                            ], 
                            width=2),

                    dbc.Col(
                        html.Div([
                            html.H6('Select number of individual IFR runs:', className="form-label"),
                            dcc.Slider(
                                id='features_per_iter_slider',
                                min=1,
                                max=n_ifr_runs,
                                step=1,
                                value=n_ifr_runs,
                                marks={str(i): str(i) for i in range(1, 
                                                                    n_ifr_runs, 
                                                                    int(n_ifr_runs / 10))},
                                tooltip={"placement": "bottom"},
                                updatemode='drag')
                            ]), 
                        width=8),

                    dbc.Col(html.Button('Randomize', 
                                        className='form-label btn btn-primary', 
                                        style={"float": 'right', 
                                                'background':self.plotly_colors['train_line'],  
                                                'border-color':self.plotly_colors['train_line']},  
                                        id='feature_trials_randomize_button'), 
                            width=2)
                ]),

                dbc.Row([
                    dbc.Col(dcc.Graph(id='number_of_feature_per_iter_figure'))
                    ], 
                    style={'padding-top':'15px'}),


                ######################################################
                ####### Feature Frequencies
                ######################################################

                dbc.Row([
                    dbc.Col(dbc.Card(html.H3(children='Feature frequencies',
                                            className="text-center"), 
                                    body=True), 
                            className="mb-4", 
                            style={'padding-top':'100px'})
                ]),

                dbc.Row([
                    dbc.Col(html.Label('Choose bin size:'), width=3),
                    dbc.Col(dcc.Slider(id='features_freq_bin_slider',
                                        min=0,
                                        max=self.results_['frequency'].max(),
                                        step=1,
                                        value=0,
                                        marks={str(i): str(i) for i in range(0, 
                                                                            self.results_['frequency'].max(), 
                                                                            int(self.results_['frequency'].max()/10))
                                                },
                                        tooltip={"placement": "bottom"},
                                        updatemode='drag'), 
                            width = 9)
                    ], 
                    style={'display': 'none'}),


                dbc.Row([
                    dbc.Col([
                            html.H6('Choose Y-axis Scale:', className="form-label"),
                            dcc.RadioItems(id='features_freq_y_axistype_radio',
                                            options=[
                                                {'label': 'Log', 'value': 'log'},
                                                {'label': 'Linear', 'value': 'linear'},
                                            ],
                                            value='log',                                                
                                            inputStyle={"margin-right": "5px", "margin-left": "15px"})
                            ], 
                            width=3),

                    dbc.Col([
                            html.H6('Frequency Cutoff:', className="form-label"),
                            dcc.Slider(
                                id='features_freq_cutoff_slider',
                                min=0,
                                max=self.results_['frequency'].max(),
                                step=1,
                                value=0,
                                marks={str(i): str(i) for i in range(0, 
                                                                    self.results_['frequency'].max(), 
                                                                    int(self.results_['frequency'].max() / 10)
                                                                    )},
                                tooltip={"placement": "bottom"},
                                updatemode='drag')
                            ], 
                            width=9),
                ]),

                dbc.Row([
                    dbc.Col(dcc.Graph(id='feature_frequency_plot'))
                    ],
                    style={'padding-top':'15px'}),

                html.Hr()
            ])
            elif tab_value == 'tab-4':
                return html.Div([
                ######################################################
                ####### Weight Vectors
                ######################################################

                dbc.Row([
                    dbc.Col(dbc.Card(html.H3(children='Weight Vectors',
                                            className="text-center"), 
                                    body=True), 
                            className="mb-4")
                ]),

                dbc.Row([
                    dbc.Col([
                        html.H6('Select an individual IFR run:', className="form-label"),
                        dcc.Slider(
                            id='weight_vector_slider',
                            min=0,
                            max=n_ifr_runs,
                            step=1,
                            value=0,
                            marks={str(i): str(i) for i in range(0, 
                                                                n_ifr_runs, 
                                                                int(n_ifr_runs / 10)
                                                                )},
                            tooltip={"placement": "bottom"},
                            updatemode='drag')
                        ]),
                    ]),

                dbc.Row([
                    dbc.Col(dcc.Graph(id='weight_vector_score_plot'))
                    ],
                    style={'padding-bottom': '0px', 'padding-top': '0px'}),


                dbc.Row([
                        dbc.Col(dcc.Graph(id='weight_ratio_plot'), 
                                width = 6, 
                                style={'padding-right':'0px'}),
                        dbc.Col(dcc.Graph(id='absolute_weight_vector_plot'), 
                                width = 6, 
                                style={'padding-left': '0px'})
                    ], 
                    style={'padding-top': '0px'}),
                
                html.Hr(),      
            ])


        @app.callback(Output('classification_scores', 'figure'),
                        Input('score_radio', 'value'),
                        Input('score_slider', 'value'),
                        Input('score_trials_randomize_button', 'n_clicks'))
        def _update_scores_plot(radio_value, slider_value, n_botton_clicks):

            train_scores = self.diagnostic_information_['train_scores']
            valid_scores = self.diagnostic_information_['validation_scores']
            n_scores = len(train_scores.columns)

            fig = go.Figure(layout = {
                                    'xaxis': {'anchor': 'y', 'domain': [0.0, 1.0], 'title': {'text': 'Iteration Index'}},
                                    'yaxis': {'anchor': 'x', 'domain': [0.0, 1.0], 'title': {'text': 'Score'}}},
                            layout_yaxis_range=[0,1.1])
            fig.update_layout(margin=dict(t=30))
                 
            max_len = len(train_scores)
            xticks = np.arange(max_len)
            fig.update(layout_xaxis_range = [-1, max_len + 2])   
            fig.add_trace(go.Scatter(x = [-5, max_len + 50], 
                                        y = [self.validation_score_cutoff, self.validation_score_cutoff],
                                        line=dict(color=self.plotly_colors['cutoff_line'], dash='dash'), 
                                        name='Cutoff',
                                        showlegend=True))

            idxs = np.arange(n_scores)
            ctx = dash.callback_context
            if ctx.triggered:
                div_id = ctx.triggered[0]['prop_id'].split('.')[0]
                if div_id == 'score_trials_randomize_button':
                    np.random.shuffle(idxs)
            
            if radio_value == 'individual':

                for i, n_iter in enumerate(idxs[:slider_value]):
                
                    fig.add_trace(go.Scatter(x = xticks, 
                                            y = train_scores[n_iter], 
                                            line_color=self.plotly_colors['train_line'],
                                            name="Train Scores",
                                            showlegend=(i == 0)))

                    fig.add_trace(go.Scatter(x = xticks, 
                                            y = valid_scores[n_iter],  
                                            line_color=self.plotly_colors['validation_line'],
                                            name='Validation Scores',
                                            showlegend=(i == 0)))
            
            elif radio_value == 'mean':

                train_mean = train_scores[idxs[:slider_value]].mean(axis=1)
                train_std = train_scores[idxs[:slider_value]].std(axis=1)
                train_lower = train_mean - train_std
                train_upper = train_mean + train_std

                fig.add_trace(go.Scatter(x = xticks , 
                                            y = train_mean,
                                            line_color=self.plotly_colors['train_line'],
                                            mode='lines+markers',
                                            name="Train Scores"))

                fig.add_trace(go.Scatter(
                    name='Train Upper Bound',
                    x=xticks,
                    y= train_upper,
                    mode='lines',
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    showlegend=False
                ))

                fig.add_trace(go.Scatter(
                    name='Train Lower Bound',
                    x=xticks,
                    y=train_lower,
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    mode='lines',
                    fillcolor=self.plotly_colors['train_fill'],
                    fill='tonexty',
                    showlegend=False
                ))

                valid_mean = valid_scores[idxs[:slider_value]].mean(axis=1)
                valid_std = valid_scores[idxs[:slider_value]].std(axis=1)
                valid_lower = valid_mean - valid_std
                valid_upper = valid_mean + valid_std

                fig.add_trace(go.Scatter(x = xticks , 
                                            y = valid_mean,
                                            line_color=self.plotly_colors['validation_line'],
                                            mode='lines+markers',
                                            name="Validation Scores"))

                fig.add_trace(go.Scatter(
                    name='Validation Upper Bound',
                    x=xticks,
                    y= valid_upper,
                    mode='lines',
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    showlegend=False
                ))

                fig.add_trace(go.Scatter(
                    name='Validation Lower Bound',
                    x=xticks,
                    y=valid_lower,
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    mode='lines',
                    fillcolor=self.plotly_colors['validation_fill'],
                    fill='tonexty',
                    showlegend=False
                ))
            
            return fig


        @app.callback(Output('number_of_feature_per_iter_figure', 'figure'),
                        Input('features_per_iter_radio', 'value'),
                        Input('features_per_iter_slider', 'value'),
                        Input('feature_trials_randomize_button', 'n_clicks'))
        def _update_features_per_iter_plot(radio_value, slider_value, n_botton_clicks):

            features = self.diagnostic_information_['true_feature_count']
            n_runs = len(features.columns)

            fig = go.Figure(layout = {
                                    'xaxis': {'anchor': 'y', 'title': {'text': 'Iteration Index'}},
                                    'yaxis': {'anchor': 'x', 'title': {'text': 'Number of Features'}}})

            fig.update_layout(margin=dict(t=30))                        
            max_len = len(features)
            xticks = np.arange(max_len)

            fig.update(layout_xaxis_range = [-1, max_len + 2])   
     
            idxs = np.arange(n_runs)
            ctx = dash.callback_context
            if ctx.triggered:
                div_id = ctx.triggered[0]['prop_id'].split('.')[0]
                if div_id == 'feature_trials_randomize_button':
                    np.random.shuffle(idxs)

            if radio_value == 'individual':
                for i, n_iter in enumerate(idxs[:slider_value]):
                
                    fig.add_trace(go.Scatter(x = xticks, 
                                            y = features[n_iter], 
                                            line_color=self.plotly_colors['validation_line'],
                                            showlegend=False))
            
            elif radio_value == 'mean':
                slider_visibility = {'display': 'none'}

                features_mean = features[idxs[:slider_value]].mean(axis=1)
                features_std = features[idxs[:slider_value]].std(axis=1)
                features_lower = features_mean - features_std
                features_upper = features_mean + features_std

                fig.add_trace(go.Scatter(x = xticks , 
                                            y = features_mean,
                                            line_color=self.plotly_colors['validation_line'],
                                            mode='lines+markers',
                                            showlegend=False))

                fig.add_trace(go.Scatter(
                    name='Train Upper Bound',
                    x=xticks,
                    y= features_upper,
                    mode='lines',
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    showlegend=False
                ))

                fig.add_trace(go.Scatter(
                    name='Train Lower Bound',
                    x=xticks,
                    y=features_lower,
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    mode='lines',
                    fillcolor=self.plotly_colors['validation_fill'],
                    fill='tonexty',
                    showlegend=False
                ))

            
            return fig


        def _create_hist(series, bins, x_axis_title, y_axis_title = 'Frequency', fig=None, yaxis_type = 'linear', legend = None):

            # if bins == 0:
            #     bins = 'auto'
            # y, xx = np.histogram(series[-series.isna()], bins=bins)
            # x = (xx[1:] + xx[:-1]) / 2
            
            freq_series = series.value_counts()
            x = freq_series.index
            x = pd.Series(x)
            y = freq_series.values
            xx = ['{:.2f}'.format(i) for i in x]

            if fig is None:
                fig = go.Figure()
                fig.update_layout(  xaxis_title=x_axis_title,
                                    title_x=0.5,
                                    xaxis=dict(
                                        tickmode='array',
                                        tickvals=xx))
                fig.update_yaxes(title=y_axis_title, type='linear' if yaxis_type == 'linear' else 'log')
                fig.update_layout(margin=dict(t=30))
            
            if legend:
                fig.add_trace(go.Bar(x=x, y=y, name=legend))
            else:
                fig.add_trace(go.Bar(x=x, y=y))
            return fig


        @app.callback(
            Output('number_of_iters_hist', 'figure'),
            Input('number_of_iters_bins', 'value'))
        def _update_number_of_iterations_hist(bins):
            return _create_hist(self.diagnostic_information_['number_of_iters_per_run'], bins, 'Number of Iterations')


        @app.callback(
            Output('validation_score_hist', 'figure'),
            Input('validation_score_hist_bins', 'value'))
        def _update_validation_scores_hist(bins):
            series = pd.Series(self.diagnostic_information_['validation_scores'].stack().values)
            return _create_hist(series, bins, 'Validation Scores')


        @app.callback(
            Output('feature_frequency_plot', 'figure'),
            Input('features_freq_cutoff_slider', 'value'),
            Input('features_freq_y_axistype_radio', 'value'),
            Input('features_freq_bin_slider', 'value'))
        def _update_feature_frequency_hist(freq_cutoff, yaxis_type, bins):
            
            x_axis_title='Frequency Classes'
            y_axis_title = 'Number of Features'

            df2 = self.results_[self.results_['frequency']>=freq_cutoff]
            fig = _create_hist(df2['frequency'], 
                bins, 
                yaxis_type = yaxis_type, 
                x_axis_title=x_axis_title, 
                y_axis_title=y_axis_title, 
                legend='Actual')

            try:
                df1 = self.frequency_cutoff_results_[self.frequency_cutoff_results_['frequency']>=freq_cutoff]
                fig = _create_hist(df1['frequency'].apply(np.round), 
                    0, 
                    fig = fig, 
                    yaxis_type = yaxis_type, 
                    x_axis_title=x_axis_title, 
                    y_axis_title=y_axis_title, 
                    legend='Null Model')

            except AttributeError:
                
                logger.error('Null model information not found!')

            return fig


        @app.callback(  
            Output('weight_vector_score_plot', 'figure'),
            Output('absolute_weight_vector_plot', 'figure'),
            Output('weight_ratio_plot', 'figure'),
            Input('weight_vector_slider', 'value'),
            Input('weight_vector_score_plot', 'clickData')
            )
        def _update_weight_vector_score_plot(slider_value, click_data):

            train_scores = self.diagnostic_information_['train_scores']
            valid_scores = self.diagnostic_information_['validation_scores']

            fig = go.Figure(layout = {
                                    'xaxis': {'anchor': 'y', 'domain': [0.0, 1.0], 'title': {'text': 'Iteration Index'}},
                                    'yaxis': {'anchor': 'x', 'domain': [0.0, 1.0], 'title': {'text': 'Score'}}},
                            layout_yaxis_range=[0,1.1])

            fig.update_layout(margin=dict(t=30, b=20), height = 300) 

            
            max_len = train_scores[slider_value].count()
            xticks = np.arange(max_len)
            fig.update(layout_xaxis_range = [-1, max_len + 2])   

            fig.add_trace(go.Scatter(x = [-5, max_len + 50], 
                                        y = [self.validation_score_cutoff, self.validation_score_cutoff],
                                        line=dict(color=self.plotly_colors['cutoff_line'], dash='dash'), 
                                        name='Cutoff',
                                        showlegend=True))
            
            fig.add_trace(go.Scatter(x = xticks, 
                                    y = train_scores[slider_value], 
                                    line_color=self.plotly_colors['train_line'],
                                    name="Train Scores", 
                                    mode='lines+markers',
                                    showlegend=True))

            fig.add_trace(go.Scatter(x = xticks, 
                                    y = valid_scores[slider_value],  
                                    line_color=self.plotly_colors['validation_line'],
                                    name='Validation Scores',
                                    mode='lines+markers',
                                    showlegend=True))
            
            if click_data:
                idx = click_data['points'][0]['pointIndex']
            else:
                idx = 0

            ctx = dash.callback_context
            if ctx.triggered:
                div_id = ctx.triggered[0]['prop_id'].split('.')[0]
                if div_id == 'weight_vector_slider':
                    idx = 0

            
            updated_score_fig = _set_marker_at_hover_point(fig, idx)

            weight_ratios = self.diagnostic_information_['weight_ratios']
            sorted_weights = self.diagnostic_information_['sorted_abs_weights']

            ratio_fig = go.Figure(layout = {
                                    'xaxis': {'anchor': 'y', 'domain': [0.0, 1.0], 'title': {'text': 'Feature Index'}},
                                    'yaxis': {'anchor': 'x', 'domain': [0.0, 1.0], 'title': {'text': 'Weight Ratios'}},
                                    'title_text': 'Weight Ratios for the current selection', 'title_x':0.5})

            ratio_fig.update_layout(margin=dict(b=20), height = 300) 
            xlim = self.diagnostic_information_['true_feature_count'][slider_value].iloc[idx]
            if xlim is np.nan:
                xlim = 0
            xlim =  + 20
            xticks = np.arange(xlim)
            ratio_fig.update(layout_xaxis_range = [-1, xlim])   

            sorted_weights_fig = copy.copy(ratio_fig)

            ratio_fig.add_trace(go.Scatter(x = xticks, 
                                    y = weight_ratios[slider_value].iloc[idx], 
                                    line_color=self.plotly_colors['cutoff_line'],
                                    name="Weight Ratios", 
                                    mode='lines+markers',
                                    showlegend=False))


            sorted_weights_fig.add_trace(go.Scatter(x = xticks, 
                                    y = sorted_weights[slider_value].iloc[idx], 
                                    line_color=self.plotly_colors['cutoff_line'],
                                    name="Sorted Absolute Weights", 
                                    mode='lines+markers',
                                    showlegend=False))

            sorted_weights_fig.update_layout({'yaxis': {'title': {'text': 'Sorted Absolute Weights'}}, 
                                            'title_text': "Sorted Absolute Weights for the current selection", 'title_x':0.5})
            return updated_score_fig, sorted_weights_fig, ratio_fig


        def _set_marker_at_hover_point(fig, idx):
            legend_printed = False
            for v in fig['data']:
                if v['name'] != 'Cutoff':
                    x = v['x'][idx]
                    y = v['y'][idx]
                    fig.add_trace(go.Scatter(x = [x], 
                                            y = [y],  
                                            line_color='rgb(92, 201, 154)',
                                            name='Current Selection',
                                            mode='markers',
                                            showlegend=legend_printed==False))
                    legend_printed=True
            return fig

        app.run_server(port=55001)
    