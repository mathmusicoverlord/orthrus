# from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import random
from calcom.solvers import LPPrimalDualPy

class IFR:


    """
    The Iterative Feature Removal algorithms extracts features for many data partitions independently and combines them and keep track of feature frequency, weights 
    and iterations in which each feature was extracted. During execution, the data is partitioned into training and validation sets 'repetition' number of times and 
    then for each partition, one feature set is extracted. So, a total of repetition * num_partitions independent feature sets are extracted and the results are 
    merged to create the output.  

    For each feature set, the algorithm can halt because of the following conditions:
        1. BSR on validation partition is below cutoff
        2. Jump does not occur in the array of sorted absolute weights
        3. Jump occurs but the weight at the jump is too small ( < 10e-6)
        4. #selected features is greater than max_features_per_iter_ratio * #samples in training partition. This condition exists to prevent overfitting.
        5. max_iters number of iterations complete successfully

    When one of these conditions happen, further feature extraction on the current fold is stopped and begins on the next partition.

    Parameters:
        repetition (int): Determines the number of times to partition the dataset. (default: 10)
        
        partition_method (string): A partition method that is compatible with calcom.utils.generate_partitions (default: 'stratified_k-fold')

        nfolds (int): The number of folds to partition data into (default: 3)

        max_iters (int): Determines the maximum number of iterations of IFR on one data partition(default: 5)
        
        cutoff (float): Threshold for the validation BSR (balanced success rate) to halt the process. (default: 0.75)

        jumpratio (float): The relative drop in the magnitude of coefficients in weight vector to identify numerically zero weights (default: 100)

        max_features_per_iter_ratio (float) = A fraction that limits the max number of features that can be extracted per iteration. (default: 0.8)
                                             if the number if selected features is greater than max_features_per_iter_ratio * #samples in training partition, further execution
                                             on the current fold is stopped. 
  
        verbosity (int) : Determines verbosity of print statments; 0 for no output; 2 for full output. (default: 0)
                
    Attributes:
        diagnostic_information_ (dict): Holds execution information for each interation of each partition.

    Return:
        results (pandas.DataFrame): The dataframe is indexed by feature_ids and contains the results of IFR. Each column contains different infromation for each feature as described below:
                frequency: How many times the feature is extracted
                weights: Contains a list of weights, from the weight vectors during training on different partitions. Each value corresponds to the weight for the feature over different extractions.
                            The length of the weights is equal to the frequency.
                selection_iteration: Contains a list of indices of the iteration when the feature was extracted over different data partitions. The length of the list is equal to the frequency.
    Examples:
            >>> import datasci.core.dataset as DS
            >>> import datasci.sparse.feature_selection.IterativeFeatureRemoval as IFR
            
            >>> x = DS.load_dataset(file_path)
            >>> ifr = IFR.IFR(
                repetition = 500,
                nfolds = 4,
                max_iters = 100,
                cutoff = .6,
                jumpratio = 5,
                max_features_per_iter_ratio = 2
                verbosity = 2,
                )

            >>> result = x.feature_select(ifr,
                        attrname,
                        selector_name='IFR',
                        f_results_handle='results',
                        append_to_meta=False,
                        )
            >>> #see feature select method for details
            >>> ifr.plot_basic_diagnostic_stats()
    """
    def __init__(self,
                repetition=10,
                partition_method = 'stratified_k-fold',
                nfolds = 3,
                max_iters=5,
                cutoff=0.75,
                jumpratio=100.,
                max_features_per_iter_ratio=0.8,
                verbosity=0):

        self.repetition = repetition   # Number of time the data is randomly partitioned.
        self.partition_method =  partition_method # Passed to calcom.utils.generate_partitions
        self.nfolds = nfolds   # Passed to calcom.utils.generate_partitions
        self.max_iters = max_iters    # Max iterations for IFR on one data partition
        self.cutoff = cutoff    # validation BSR threshold
        self.jumpratio = jumpratio # Relative drop needed to detect numerically zero weights in SSVM.
        self.max_features_per_iter_ratio = max_features_per_iter_ratio   # fraction of training data samples as cutoff for maximum features extracted per iteration 
        self.verbosity = verbosity    # Verbosity of print statements; make positive to see detail.
        self.diagnostic_information_ = {}
        self._diagnostic_information_keys = ['train_bsrs', 'validation_bsrs', 'sorted_abs_weights', 'weight_ratios',
                                            'features', 'true_feature_count', 'cap_breached']
        self._initialize_diagnostic_dictionary(self.diagnostic_information_)
        self.diagnostic_information_['exit_reasons'] = []
        super(IFR, self).__init__()
    #

    def _initialize_diagnostic_dictionary(self, diag_dict):
        for key in self._diagnostic_information_keys:
            diag_dict[key] = []
       
    def _add_diagnostic_info_for_current_iteration(self, diag_dict, train_bsr, validation_bsr,
        sorted_abs_weights, weight_ratios, features, true_feature_count, cap_breached):

        diag_dict.get('train_bsrs', []).append(train_bsr)
        diag_dict.get('validation_bsrs', []).append(validation_bsr)
        diag_dict.get('sorted_abs_weights', []).append(sorted_abs_weights)
        diag_dict.get('weight_ratios', []).append(weight_ratios)
        diag_dict.get('features', []).append(features)
        diag_dict.get('true_feature_count', []).append(true_feature_count)
        diag_dict.get('cap_breached', []).append(cap_breached)


    def _sanity_check_diagnostics(self, diag_dict, n_iters):
        arr = np.zeros(len(self._diagnostic_information_keys))
        for i, key in enumerate(self._diagnostic_information_keys):
            arr[i] = len(diag_dict[key])

        assert np.unique(arr).shape[0] == 1, 'Lenghts of lists for the diagnostic information do not match. They should be of same size' 
        if n_iters > 1:
            assert np.unique(arr)[0] == n_iters, 'diagnostic dictionary does not contain all the information'

    def _add_diagnostic_info_for_data_partition(self, diag_dict, n_data_partition, exit_reason):
        for key in self._diagnostic_information_keys:
            self.diagnostic_information_[key].append(diag_dict[key])
        
        self.diagnostic_information_['exit_reasons'].append(exit_reason)

    
    def _initialize_results(self, n_attributes):
        '''
        Initializes self.results attribute
        '''
        self.results = pd.DataFrame(index=np.arange(n_attributes))
        self.results['frequency'] = 0
        self.results['weights'] = np.empty((len(self.results), 0)).tolist() 
        self.results['selection_iteration'] = np.empty((len(self.results), 0)).tolist() 


    def _update_frequency_in_results(self, features):
        '''
        Increments the values of features by 1, for passed features, in frequency column of self.results
        '''
        self.results.loc[features, 'frequency'] =  self.results.loc[features, 'frequency'] + 1


    def _update_selection_iteration_in_results(self, features, iteration):
        '''
        Appends the iteration number, for passed features, in selection_iteration column of self.results
        '''
        for feature in features:
            iter_list = self.results.loc[feature, 'selection_iteration']
            iter_list.append(iteration)

    def _update_weights_in_results(self, features, weights):
        '''
        Appends the weight for passed features in the weights column in self.results
        ''' 
        #iterate over features
        for feature, weight in zip(features, weights):
            #append the weight to the list of weights for the current feature
            weights = self.results.loc[feature, 'weights']
            weights.append(weight)
            


    def fit(self, data, labels):
        '''
        data        : m-by-n array of data, with m the number of observations.
        labels      : m vector of labels for the data

        return      : a dictionary of features {keys=feature : value=no_of_time_it_was_selected}
        '''
        import calcom
        import numpy as np
        import time

        m,n = np.shape(data)

        if self.verbosity>0:
            print('IFR parameters:\n')
            print('repetition', self.repetition)
            print('partition_method', self.partition_method)
            print('nfolds', self.nfolds)
            print('max_iters', self.max_iters)
            print('jumpratio', self.jumpratio)
            print('cutoff', self.cutoff)
            print('max_features_per_iter_ratio', self.max_features_per_iter_ratio)
            print('verbosity', self.verbosity)
            print('\n')
        #


        if self.nfolds < 2:
            raise ValueError("Number of folds have to be greater than 1")


        self._initialize_results(n)

        bsr = calcom.metrics.ConfusionMatrix('bsr')

        # start processing

        #total time is the time taken by this method divided by total Number
        #of iterations it will run for.
        total_start_time = time.time()

        #method time is similar to total time, but calculates just the
        #time taken by ssvm to fit the data
        total_method_time = 0

        total_iterations = 0
        n_data_partition = 0
        for n_rep in range(self.repetition):

            if self.verbosity>0:
                print("=====================================================")
                print("beginning of repetition ", n_rep+1, " of ", self.repetition)
                print("=====================================================")


            partitions = calcom.utils.generate_partitions(labels, method=self.partition_method, nfolds=self.nfolds)

            for i, partition in enumerate(partitions):

                n_data_partition +=1

                if self.verbosity>0:
                    print("=====================================================")
                    print("beginning of execution for fold ", i+1, " of ", len(partitions))
                    print("=====================================================")
                #

                list_of_features_for_curr_fold = np.array([], dtype=np.int64)
                list_of_weights_for_curr_fold = np.array([], dtype=np.int64)            
                selected = np.array([], dtype=np.int64)
                # Mask array which tracks features which haven't been removed.
                active_mask = np.ones(n, dtype=bool)

                train_idx, validation_idx = partition
                train_data = data[train_idx, :]
                train_labels = labels[train_idx]

                validation_data = data[validation_idx, :]
                validation_labels = labels[validation_idx]

                #create an empty dictionary to store diagnostic info for the
                #current data partition, this dictionary has info about each iteration
                #on the current data partition

                diagnostic_info_dictionary = {}
                self._initialize_diagnostic_dictionary(diagnostic_info_dictionary)
                exit_reason = "max_iters"
                for i in range(self.max_iters):
                    total_iterations += 1
                    if self.verbosity > 1:
                        print("=====================================================")
                        print("beginning of inner loop iteration ", i+1)
                        print("Number of features selected for this fold: %i of %i"%(len(list_of_features_for_curr_fold),n))
                        print("Checking BSR of complementary problem... ",end="")
                    #

                    #redefine SSVM classifier with new random parameters
                    ssvm = calcom.classifiers.SSVMClassifier()
                    ssvm.params['C'] = 1.
                    ssvm.params['use_cuda'] = True
                    ssvm.params['method'] = LPPrimalDualPy

                    tr_d = np.array( train_data[:,active_mask] )
                    te_d = np.array( validation_data[:,active_mask] )
                    #import pdb; pdb.set_trace()
                    try:
                        method_start_time = time.time()
                        ssvm.fit(tr_d, train_labels)
                        total_method_time += time.time() - method_start_time
                    except Exception as e:
                        if self.verbosity>0:
                            print("Warning: during the training process the following exception occurred:\n")
                            print(str(e))
                            print("\nBreaking the execution for the current data fold")
                            #save the diagnostic information
                            
                        self._add_diagnostic_info_for_current_iteration(diagnostic_info_dictionary,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None)
                        exit_reason = "exception_in_ssvm_fitting"
                        break
                    
                    weight = ssvm.results['weight']

                    #calculate BSR for training data
                    pred_train = ssvm.predict(tr_d)
                    bsrval_train = bsr.evaluate(train_labels, pred_train)
                    if self.verbosity>1:
                        print('')
                        print("Training BSR %.3f. "%bsrval_train)
                        print("")

                    #calculate BSR for validation data
                    pred_validation = ssvm.predict(te_d)
                    bsrval_validation = bsr.evaluate(validation_labels, pred_validation)

                    if self.verbosity>1:
                        print("Validation BSR %.3f. "%bsrval_validation)
                        print("")

                    #Check if BSR is above cutoff
                    if (bsrval_validation < self.cutoff):
                        if self.verbosity>1:
                            print("BSR below cutoff, exiting inner loop.")

                        #save the diagnostic information for this iteration
                        #in this case we only have train and validation bsr
                        self._add_diagnostic_info_for_current_iteration(diagnostic_info_dictionary,
                            bsrval_train,
                            bsrval_validation,
                            None,
                            None,
                            None,
                            None,
                            None)

                        #break out of current loop if bsr is below cutoff
                        exit_reason = "validation_bsr_cutoff"
                        break

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


                    #check if sufficient jump was found
                    if len(jumpidxs)==0:
                        #jump never happened.
                        #save the diagnostic information for this iteration
                        #we still do not have the selected feature count and features

                        self._add_diagnostic_info_for_current_iteration(diagnostic_info_dictionary,
                            bsrval_train,
                            bsrval_validation,
                            sorted_abs_weights,
                            weight_ratios,
                            None,
                            None,
                            None)
                        exit_reason = "jump_failed"
                   
                   
                        #break out of the loop
                        if self.verbosity>1:
                            print('There was no jump of sufficient size between ratios of successive coefficients in the weight vector.')
                            print("Discarding iteration..")
                        break

                    else:
                        count = jumpidxs[0]

                    #check if the weight at the jump is greater than cutoff
                    if sorted_abs_weights[count] < 1e-6:

                        self._add_diagnostic_info_for_current_iteration(diagnostic_info_dictionary,
                            bsrval_train,
                            bsrval_validation,
                            sorted_abs_weights,
                            weight_ratios,
                            None,
                            None,
                            None)
                        exit_reason = "small_weight_at_jump"
                        if self.verbosity>1:
                            print('Weight at the jump(', sorted_abs_weights[count] ,')  smaller than weight cutoff(1e-6).')
                            print("Discarding iteration..")
                        break

                    count += 1
                    cap_breached = False
                    #check if the number of selected features is greater than the cap
                    if count > int(self.max_features_per_iter_ratio * train_data.shape[0]):

                        self._add_diagnostic_info_for_current_iteration(diagnostic_info_dictionary,
                            bsrval_train,
                            bsrval_validation,
                            sorted_abs_weights,
                            weight_ratios,
                            None,
                            None,
                            None)
                        exit_reason = "max_features_per_iter_breached"
                        if self.verbosity>1:
                            print('More features selected than the ', self.max_features_per_iter_ratio, ' ratio of training data samples(', train_data.shape[0], ')')
                            print("Discarding iteration..")
                        
                        break

                    
                    #select features: order is list of sorted features
                    selected = order[:count]

                    if self.verbosity>1:
                        print("\nSelected features on this iteration:")
                        print(selected)
                        print("\n")
                    #

                    # Selected indices are relative to the current active set.
                    # Get the mapping back to the original indices.
                    active_idxs = np.where(active_mask)[0]

                    active_mask[active_idxs[selected]] = 0

                    #append the selected features to the list_of_features_for_curr_fold
                    list_of_features_for_curr_fold = np.concatenate([list_of_features_for_curr_fold ,  active_idxs[selected]])
                    list_of_weights_for_curr_fold = np.concatenate([list_of_weights_for_curr_fold ,  weight.flatten()[order][:count]])
                    #save the diagnostic information for this iteration
                    #here we have all the information we need
                    self._add_diagnostic_info_for_current_iteration(diagnostic_info_dictionary,
                        bsrval_train,
                        bsrval_validation,
                        sorted_abs_weights,
                        weight_ratios,
                        active_idxs[selected],
                        count,
                        cap_breached)

                    if self.verbosity>1:
                        print('Removing %i features from training and validation matrices.'%len(selected))
                        print("\n")

                    #update the selection_iteration for the features selected in the current iteration
                    self._update_selection_iteration_in_results(active_idxs[selected], i+1)    

                # update the feature set dictionary based on the features collected for current fold
                self._update_frequency_in_results(list_of_features_for_curr_fold)
                self._update_weights_in_results(list_of_features_for_curr_fold, list_of_weights_for_curr_fold) 
                #save the diagnostic information for this data partition
                self._sanity_check_diagnostics(diagnostic_info_dictionary, i+1)
                self._add_diagnostic_info_for_data_partition(diagnostic_info_dictionary, n_data_partition, exit_reason)

            #
        total_time_per_iteration = (time.time() - total_start_time) / total_iterations
        method_time_per_iteration = total_method_time / total_iterations
        self.total_time = total_time_per_iteration
        self.method_time = method_time_per_iteration

        if self.verbosity>0:
            print("=====================================================")
            print("Finishing Execution. %d features out of a total of %d features were selected."% ((self.results['frequency'] > 0).sum(), data.shape[1]))
            print("=====================================================")

        return self.results
    #


    def plot_basic_diagnostic_stats(self, validation_bsr_iteration_idx = None, n_random_exp = -1):
        exit_reasons = self.diagnostic_information_['exit_reasons']
        #self.diagnostic_information_.pop('exit_reasons')

        fig, axs = plt.subplots(figsize=(12, 4), nrows = 2, ncols = 3)
        n_elements = len(self.diagnostic_information_['validation_bsrs'])
        idx = np.arange(n_elements)
        random.shuffle(idx)

        if n_random_exp != -1 and n_random_exp < idx.shape[0]:
            idx = idx[:n_random_exp]

        axs[0][0].set_title('validation BSR')
        axs[0][0].set_ylim(0, 1.1)
        axs[0][1].set_title('Features Selected')
        max_iters = 0
        for j in idx:    
            num_iters = len(self.diagnostic_information_['validation_bsrs'][j])
            if num_iters == 1:
                axs[0][0].plot(0, self.diagnostic_information_['validation_bsrs'][j], marker='.', alpha = 0.5)
            else:
                axs[0][0].plot(self.diagnostic_information_['validation_bsrs'][j], alpha = 0.5)
            #axs[i, 0].plot(0, dict['cutoff'], len(dict['validation_bsr']), dict['cutoff'])
            if max_iters < num_iters:
                max_iters = num_iters

            axs[0][1].plot(self.diagnostic_information_['true_feature_count'][j], alpha = 0.5)
            #axs[i, 1].plot(0, dict['max_features_per_iter_ratio'], len(dict['validation_bsr']), dict['max_features_per_iter_ratio'])

        axs[0][0].set_xlim(-.9, max_iters + 1)
        exit_r = np.array(exit_reasons)
        n_elements = np.unique(exit_r).shape[0]
        
        axs[0][2].hist(exit_reasons, n_elements, histtype='stepfilled', facecolor='g', alpha=0.75)
        axs[0][2].set_title('Exit Reasons')


        #second row of plots

        #first plot shows the histogram of number of iterations
        num_iterations = np.array([len(x) for x in self.diagnostic_information_['true_feature_count']])
        axs[1][0].hist(num_iterations)
        axs[1][0].set_ylabel('Frequency')
        axs[1][0].set_xlabel('# Iterations per fold')

        #second plot show the histogram of validation bsrs
        if validation_bsr_iteration_idx == None:
            validation_bsrs = [item for sublist in self.diagnostic_information_['validation_bsrs'] for item in sublist]
            label = 'Validation BSRS over all iterations and folds'
        else:
            validation_bsrs = []
            for sublist in self.diagnostic_information_['validation_bsrs']:
                if len(sublist) >= validation_bsr_iteration_idx:
                    validation_bsrs = validation_bsrs.append(sublist[validation_bsr_iteration_idx])
            label = 'Validation BSRS over iteration# %d of all folds'%validation_bsr_iteration_idx
        axs[1][1].hist(validation_bsrs)
        axs[1][1].set_ylabel('Frequency')
        axs[1][1].set_xlabel(label)

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.show()

#


