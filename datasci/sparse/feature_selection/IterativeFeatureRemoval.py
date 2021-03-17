# from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import random
import pickle

class IFR:

    def __init__(self):
        '''
        Default parameters:

        'cutoff'    : Threshold for the BSR (balanced success rate) to halt the process. (default: 0.75)
        'jumpratio' : The relative drop in the magnitude of coefficients in weight vector to identify numerically zero weights (default: 100)
        'verbosity' : Determines verbosity of print statments; 0 for no output; 2 for full output. (default: 0)
        'nfolds'    : The number of folds to partition data into (default: 3)
        'repetition': Determines the number of times to repeat the Feature Removal. the algorithm runs for repetitions * nfolds iterations (default: 3)
        'max_iters' : Determines the maximum number of iteration of IFR on a particular data partition fold(default: 5)
        'max_features_per_iter': Determines the maximum number of features selected for one iteration on a particular data partition fold (default: 50)
        'C'         : The value for the sparsity promoting parameter used in SSVM (Sparse SVM). (default: 1.)


        '''
        from calcom.solvers import LPPrimalDualPy

        self.params = {}

        self.params['partition_method'] = 'stratified_k-fold'   # Passed to calcom.utils.generate_partitions
        self.params['nfolds'] = 3   # Passed to calcom.utils.generate_partitions
        self.params['max_iters'] = 5    # Max iterations for IFR on one data partition
        self.params['cutoff'] = 0.75    # BSR threshold
        self.params['jumpratio'] = 100. # Relative drop needed to detect numerically zero weights in SSVM.
        self.params['repetition'] = 3   # Number of repetitions to do.
        self.params['max_features_per_iter_ratio'] = 0.8   # fraction of training data samples as cutoff for maximum features extracted per iteration 
        self.params['method'] = LPPrimalDualPy   # Default linear problem solver for SSVM
        self.params['use_cuda'] = False # flag to run SSVM on GPU
        self.params['C'] = 1.           # Sparsity promoting parameter for use in SSVM
        self.params['verbosity'] = 0    # Verbosity of print statements; make positive to see detail.
        self.diagnostic_information = {}
        self._diagnostic_information_keys = ['train_bsrs', 'test_bsrs', 'sorted_abs_weights', 'weight_ratios',
                                            'features', 'true_feature_count', 'cap_breached']
        self._initialize_diagnostic_dictionary(self.diagnostic_information)
        self.diagnostic_information['exit_reasons'] = []
        super(IFR, self).__init__()
    #

    def _initialize_diagnostic_dictionary(self, diag_dict):
        for key in self._diagnostic_information_keys:
            diag_dict[key] = []
       
    def _add_diagnostic_info_for_current_iteration(self, diag_dict, train_bsr, test_bsr,
        sorted_abs_weights, weight_ratios, features, true_feature_count, cap_breached):

        diag_dict.get('train_bsrs', []).append(train_bsr)
        diag_dict.get('test_bsrs', []).append(test_bsr)
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
            self.diagnostic_information[key].append(diag_dict[key])
        
        self.diagnostic_information['exit_reasons'].append(exit_reason)

    
    def _initialize_results(self, n_attributes):
        '''
        Initializes self.results attribute
        '''
        self.results = pd.DataFrame(index=np.arange(n_attributes))
        self.results['frequency'] = 0
        self.results['weights'] = np.empty((len(self.results), 0)).tolist() 


    def _update_frequency_in_results(self, features):
        '''
        Increments the values of features by 1, for passed features, in frequency column of self.results
        '''
        self.results.loc[features, 'frequency'] =  self.results.loc[features, 'frequency'] + 1

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

        if self.params['verbosity']>0:
            print('IFR parameters:\n')
            for k,v in self.params.items():
                print( '%20s : %s'%(k,v) )
            print('\n')
        #


        if self.params['nfolds'] < 2:
            raise ValueError("Number of folds have to be greater than 1")


        self._initialize_results(n)

        bsr = calcom.metrics.ConfusionMatrix('bsr')

        #define SSVM classifier
        ssvm = calcom.classifiers.SSVMClassifier()
        ssvm.params['C'] = self.params['C']
        ssvm.params['use_cuda'] = self.params['use_cuda']
        ssvm.params['method'] = self.params['method']
        # start processing

        #total time is the time taken by this method divided by total Number
        #of iterations it will run for.
        total_start_time = time.time()

        #method time is similar to total time, but calculates just the
        #time taken by ssvm to fit the data
        total_method_time = 0

        total_iterations = 0
        n_data_partition = 0
        for n_rep in range(self.params['repetition']):

            if self.params['verbosity']>0:
                print("=====================================================")
                print("beginning of repetition ", n_rep+1, " of ", self.params['repetition'])
                print("=====================================================")


            partitions = calcom.utils.generate_partitions(labels, method=self.params['partition_method'], nfolds=self.params['nfolds'])

            for i, partition in enumerate(partitions):

                n_data_partition +=1
                n_inner_itr = 0
                n_jump_failed = 0

                if self.params['verbosity']>0:
                    print("=====================================================")
                    print("beginning of execution for fold ", i+1, " of ", len(partitions))
                    print("=====================================================")
                #

                list_of_features_for_curr_fold = np.array([], dtype=np.int64)
                list_of_weights_for_curr_fold = np.array([], dtype=np.int64)            
                selected = np.array([], dtype=np.int64)
                # Mask array which tracks features which haven't been removed.
                active_mask = np.ones(n, dtype=bool)

                train_idx, test_idx = partition
                train_data = data[train_idx, :]
                train_labels = labels[train_idx]

                test_data = data[test_idx, :]
                test_labels = labels[test_idx]

		        #create an empty dictionary to store diagnostic info for the
                #current data partition, this dictionary has info about each iteration
                #on the current data partition

                diagnostic_info_dictionary = {}
                self._initialize_diagnostic_dictionary(diagnostic_info_dictionary)
                exit_reason = "max_iters"
                for i in range(self.params['max_iters']):
                    n_inner_itr += 1
                    total_iterations += 1
                    if self.params['verbosity'] > 1:
                        print("=====================================================")
                        print("beginning of inner loop iteration ", n_inner_itr)
                        print("Number of features selected for this fold: %i of %i"%(len(list_of_features_for_curr_fold),n))
                        print("Checking BSR of complementary problem... ",end="")
                    #

                    #redefine SSVM classifier with new random parameters
                    ssvm = calcom.classifiers.SSVMClassifier()
                    ssvm.params['C'] = self.params['C']
                    ssvm.params['use_cuda'] = self.params['use_cuda']
                    ssvm.params['method'] = self.params['method']

                    tr_d = np.array( train_data[:,active_mask] )
                    te_d = np.array( test_data[:,active_mask] )
                    #import pdb; pdb.set_trace()
                    try:
                        method_start_time = time.time()
                        ssvm.fit(tr_d, train_labels)
                        total_method_time += time.time() - method_start_time
                    except Exception as e:
                        if self.params['verbosity']>0:
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
                    if self.params['verbosity']>1:
                        print('')
                        print("Training BSR %.3f. "%bsrval_train)
                        print("")

                    #calculate BSR for test data
                    pred_test = ssvm.predict(te_d)
                    bsrval_test = bsr.evaluate(test_labels, pred_test)

                    if self.params['verbosity']>1:
                        print("Testing BSR %.3f. "%bsrval_test)
                        print("")

                    #Check if BSR is above cutoff
                    if (bsrval_test < self.params['cutoff']):
                        if self.params['verbosity']>1:
                            print("BSR below cutoff, exiting inner loop.")

                        #save the diagnostic information for this iteration
                        #in this case we only have train and test bsr
                        self._add_diagnostic_info_for_current_iteration(diagnostic_info_dictionary,
                            bsrval_train,
                            bsrval_test,
                            None,
                            None,
                            None,
                            None,
                            None)

                        #break out of current loop if bsr is below cutoff
                        exit_reason = "test_bsr_cutoff"
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
                    jumpidxs = np.where(weight_ratios > self.params['jumpratio'])[0]


                    #check if sufficient jump was found
                    if len(jumpidxs)==0:
                        #jump never happened.
                        #save the diagnostic information for this iteration
                        #we still do not have the selected feature count and features

                        self._add_diagnostic_info_for_current_iteration(diagnostic_info_dictionary,
                            bsrval_train,
                            bsrval_test,
                            sorted_abs_weights,
                            weight_ratios,
                            None,
                            None,
                            None)
                        exit_reason = "jump_failed"
                   
                   
                        #break out of the loop
                        if self.params['verbosity']>1:
                            print('There was no jump of sufficient size between ratios of successive coefficients in the weight vector.')
                            print("Discarding iteration..")
                        break

                    else:
                        count = jumpidxs[0]

                    #check if the weight at the jump is greater than cutoff
                    if sorted_abs_weights[count] < 1e-6:

                        self._add_diagnostic_info_for_current_iteration(diagnostic_info_dictionary,
                            bsrval_train,
                            bsrval_test,
                            sorted_abs_weights,
                            weight_ratios,
                            None,
                            None,
                            None)
                        exit_reason = "small_weight_at_jump"
                        if self.params['verbosity']>1:
                            print('Weight at the jump(', sorted_abs_weights[count] ,')  smaller than weight cutoff(1e-6).')
                            print("Discarding iteration..")
                        break

                    count += 1
                    cap_breached = False
                    #check if the number of selected features is greater than the cap
                    if count > int(self.params['max_features_per_iter_ratio'] * train_data.shape[0]):

                        self._add_diagnostic_info_for_current_iteration(diagnostic_info_dictionary,
                            bsrval_train,
                            bsrval_test,
                            sorted_abs_weights,
                            weight_ratios,
                            None,
                            None,
                            None)
                        exit_reason = "max_features_reached"
                        if self.params['verbosity']>1:
                            print('More features selected than the ', self.params['max_features_per_iter_ratio'], ' ratio of training data samples(', train_data.shape[0], ')')
                            print("Discarding iteration..")
                        
                        break

                    
                    #select features: order is list of sorted features
                    selected = order[:count]

                    if self.params['verbosity']>1:
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
                        bsrval_test,
                        sorted_abs_weights,
                        weight_ratios,
                        active_idxs[selected],
                        count,
                        cap_breached)

                    if self.params['verbosity']>1:
                        print('Removing %i features from training and test matrices.'%len(selected))
                        print("\n")

                # update the feature set dictionary based on the features collected for current fold
                self._update_frequency_in_results(list_of_features_for_curr_fold)
                self._update_weights_in_results(list_of_features_for_curr_fold, list_of_weights_for_curr_fold) 
                #save the diagnostic information for this data partition
                self._sanity_check_diagnostics(diagnostic_info_dictionary, n_inner_itr)
                self._add_diagnostic_info_for_data_partition(diagnostic_info_dictionary, n_data_partition, exit_reason)

            #
        total_time_per_iteration = (time.time() - total_start_time) / total_iterations
        method_time_per_iteration = total_method_time / total_iterations
        self.params['total_time'] = total_time_per_iteration
        self.params['method_time'] = method_time_per_iteration

        if self.params['verbosity']>0:
            print("=====================================================")
            print("Finishing Execution. %d features out of a total of %d features were selected."% (self.results['frequency'] > 0).sum(), data.shape[1])
            print("=====================================================")

        return self.results
    #


    def plot_basic_diagnostic_stats(self, n_random_exp = -1):
        exit_reasons = self.diagnostic_information['exit_reasons']
        #self.diagnostic_information.pop('exit_reasons')

        fig, axs = plt.subplots(figsize=(12, 4), nrows = 1, ncols = 3)
        n_elements = len(self.diagnostic_information['test_bsrs'])
        idx = np.arange(n_elements)
        random.shuffle(idx)

        if n_random_exp != -1 and n_random_exp < idx.shape[0]:
            idx = idx[:n_random_exp]

        for j in idx:    

            axs[0].plot(self.diagnostic_information['test_bsrs'][j], alpha = 0.5)
            #axs[i, 0].plot(0, dict['cutoff'], len(dict['test_bsr']), dict['cutoff'])
            axs[0].set_title('Test BSR')
            axs[0].set_ylim(0, 1.1)


            axs[1].plot(self.diagnostic_information['true_feature_count'][j], alpha = 0.5)
            #axs[i, 1].plot(0, dict['max_features_per_iter_ratio'], len(dict['test_bsr']), dict['max_features_per_iter_ratio'])
            axs[1].set_title('Features Selected')

        exit_r = np.array(exit_reasons)
        n_elements = np.unique(exit_r).shape[0]
        
        axs[2].hist(exit_reasons, n_elements, density=True, histtype='stepfilled', facecolor='g', alpha=0.75)
        axs[2].set_title('Exit Reasons')

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.show()

    def save(self, file_path):
        """
        This method saves an instance of a ItertiveFeatureRemoval class in pickle format. 

        Args:
            file_path (str): Path of the file to save the instance to.

        Returns:
            inplace method.

        """
        # pickle class instance
        with open(file_path, 'wb') as f:
            pickle.dump(self, file=f)


# functions
def load_ifr_instance(file_path: str):
    """
    This function loads and returns an instance of a IterativeFeatureRemoval class in pickle format.

    Args:
        file_path (str): Path of the file to load the instance from.

    Returns:
        IterativeFeatureRemoval instance : Class instance encoded by pickle binary file_path.

    Examples:
            >>> ifr = load_ifr_instance(file_path='./tol_vs_res_liver_ifr.pickle')
    """
    # open file and unpickle
    with open(file_path, 'rb') as f:
        return pickle.load(f)
#

