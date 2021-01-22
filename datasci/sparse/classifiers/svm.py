from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class SSVMClassifier(BaseEstimator, ClassifierMixin):
    '''

    A class implementing a sparse support vector machine (SSVM) algorithm,
    which solves the *sparse* (l1) support vector problem. The
    primal form of the optimization problem is (?)

    min ||w||_1 + C*sum(xi_j)

        s.t. y_j*(w'*x_j - b) <= 1-xi_j,

    for points x_j with corresponding labels y_j (assumed +/-1 in this formulation),
    optimizing for vector w (the ``weight" vector) and scalar b (the ``bias"),
    with a corresponding linear (affine) model function f(x) = w'*x - b which
    approximates the original X, and classifies points using
    sign(f(x)).

    The sklearn adaptation was implemented by Eric Kehoe, CSU 2021.
    The original Python implementation of this code was by Tomojit Ghosh,
    which was originally an adaptation of a Matlab code by Sofya Chepushtanova and
    Michael Kirby.

    Parameters:
        C (float): C is the tuning parameter which weights how heavily to penalize the average error for points
            to violate the hard-margin hyperplane constraints. For values C close to zero, points will be able to
            violate the hard-margin constraints more so, and sparsity in the w vector will be maximized. Default value
            is 1.0.
            
        tol (float): Error tolerance to use in interior point method specified by `solver`. Default value is .001.

        solver (object): Solver to use for solving the above linear program.

        errorTrace (object): Not known, ignore.

        use_cuda (bool): Flag indicating whether or not to perform linear algebra operations, including solving the
            above LP, on the GPU. `use_cuda = True` means use the GPU. `use_cuda = False` is default.

        verbosity (int): Specifies the level of text output to the user. The default value in 0; indicating minimal
            text output.

        debug (bool): Passed to the solver to print debug information. Default value is False.
        
    Attributes:
        weights_ (ndarray of shape (n_features,)): Vector of weights, obtained by fitting the SSVM classifier
        via :py:func:`SSVM.fit`, defining the normal vector to the separating hyperplane.

        bias_ (float): Affine shift of the hyperplane obtained by fitting the SSVM classifier via :py:func:`SSVM.fit`.

        pred_labels_ (list of length n_samples): Prediction labels of test data obtained by predicting with the SSVM
            classifier via :py:func:`SSVM.predict

        classes_ (ndarray of shape (n_classes,)): The class labels.
    '''

    def __init__(self,
                 C: float = 1.0,
                 tol: float = 0.001,
                 solver: object = None,
                 errorTrace: object =None,
                 use_cuda: bool = False,
                 verbosity: int = 0,
                 debug: bool = False):

        # Solver parameters
        self.C = C                              # the margin weight
        self.tol = tol                          # error tolerance for interior point solver
        self.solver = solver                    # solver for solving the LP
        self.errorTrace = errorTrace
        self.use_cuda = use_cuda                # Flag to attempt to use CUDA.
        self.verbosity = verbosity              # Level of verbosity
        self.debug = debug

        # Solver attributes
        self.weights_ = None
        self.bias_ = None
        self.pred_labels_ = None
        self.classes_ = None

    def fit(self, X, y):
        '''
        Fit/training step for Sparse Support Vector Machines (SSVM). A model function

        f(x)=w'*x - b

        is found for vector w=len(x) and scalar b which optimally classify the
        training X, in the sense of solving the L1 minimization problem

        min ||w||_1 + C*sum( xi_j )
            s.t. y_j*(w'*x_j -b) <= 1-xi_j, j=1,...,n,

        where x_j are vector input X, y_j are {-1,+1} class labels for each x_j,
        xi_j are scalar slack variables.

        This code only supports binary classification right now.

        The weight vector w and bias b are stored in self.weights_ and
        self.bias_ respectively.
        '''

        import numpy as np
        try:
            import torch
        except ImportError:
            torch = None

        # Check that the stars have aligned so that we can use CUDA.
        use_cuda = self.use_cuda and torch and torch.cuda.is_available()
        if self.verbosity>0:
            if self.use_cuda and not use_cuda:
                print('PyTorch could not be imported, or could not access the GPU. Falling back to numpy implementation.')
        #

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        if len(self.classes_) != 2:
            raise ValueError("The supplied training X has fewer or greater than two labels.\nOnly binary classification is supported.")

        # Need an extra step here - SSVM wants labels -1 and 1.
        labelDict = {self.classes_[0]: -1, self.classes_[1]: 1}
        internalLabels = [labelDict[sample] for sample in y]

        nSamples = np.shape(X)[0]

        inputDim=np.shape(X)[1]
        IP=np.diag(np.ones(nSamples)).astype(int)
        eP=np.ones(nSamples).reshape(-1, 1)
        eDim=np.ones(inputDim).reshape(-1, 1)

        D=np.diag(internalLabels)    #Diagonal matrix of labels

        if use_cuda:
            D_c = torch.from_numpy(D).double().cuda()
            trData_c = torch.from_numpy(X).double().cuda()
            DX = torch.mm(D_c,trData_c).cpu().numpy()

            eP_c = torch.from_numpy(eP).double().cuda()
            De = torch.mm(D_c,eP_c).cpu().numpy()
        else:
            DX = np.dot(D, X)
            De = np.dot(D, eP)
        #

        A = np.hstack((DX, -DX, -De, De, IP))
        c = np.vstack((eDim, eDim, np.array([0]).reshape(-1, 1), np.array([0]).reshape(-1, 1), self.C*eP))

        x = self.solver(-c,-A,-eP, output_flag=0, use_cuda=use_cuda, verbosity=self.verbosity, debug=self.debug)

        self.weights_ = x[:inputDim] - x[inputDim:2*inputDim]
        self.bias_ = x[2*inputDim]-x[2*inputDim+1]

        return self
    #

    def predict(self, X):
        '''
        Classification step for Sparse Support Vector Machine (SSVM).
        After the fit/training step, vectors w and b are found to
        optimally classify the training X (in the sense described
        in the fit() docstring). New X is classified using

        sign(f(x)) = sign( w'*x - b ).
        '''
        import numpy as np
        try:
            import torch
        except ImportError:
            torch = None

        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Check that the stars have aligned so that we can use CUDA.
        use_cuda = self.use_cuda and torch and torch.cuda.is_available()
        if self.verbosity > 0:
            if self.use_cuda and not use_cuda:
                print('PyTorch could not be imported, or could not access the GPU. Falling back to numpy implementation.')

        b = self.bias_
        w = self.weights_

        if use_cuda:
            data_c = torch.from_numpy(X).double().cuda()
            w_c = torch.from_numpy(w).double().cuda()
            b_c = torch.from_numpy(b).double().cuda()
            d = torch.addmm(-1,b_c,data_c,w_c).cpu().numpy()
        else:
            d = np.dot(X, w) - b

        predicted = np.sign(d)
        predicted = np.array(predicted, dtype=int).flatten()  # can't be too sure

        # map -1, 1 labels to original labels
        invLabelDict = {-1: self.classes_[0], 1: self.classes_[1]}
        pred_labels = [invLabelDict[sample] for sample in predicted]
        self.pred_labels_ = pred_labels

        return pred_labels

    def decision_function(self, X):
        import numpy as np
        try:
            import torch
        except ImportError:
            torch = None
        #

        b = self.bias_
        w = self.weights_

        if torch and self.use_cuda:
            data_c = torch.from_numpy(X).double().cuda()
            w_c = torch.from_numpy(w).double().cuda()
            b_c = torch.from_numpy(b).double().cuda()
            d = torch.addmm(-1, b_c, data_c, w_c).cpu().numpy()
        else:
            d = np.dot(X, w) - b
        #
        return d
