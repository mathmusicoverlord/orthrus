import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.pairwise import pairwise_kernels
import torch as tc
import numpy as np
from matplotlib import pyplot as plt
from orthrus.solvers.linear import LPNewton
from orthrus.solvers.nonlinear import LPPrimalDualPy
from orthrus.sparse.feature_selection.helper import get_correlates
from copy import copy
import ray

class SSVMClassifier(BaseEstimator, ClassifierMixin):
    '''

    A class implementing a sparse support vector machine (SSVM) algorithm,
    which solves the sparse (l1) support vector problem. The
    primal form of the optimization problem is (?)

    min ||w||_1 + C\*sum(xi_j) s.t. y_j\*(w'\*x_j - b) <= 1-xi_j,

    for points x_j with corresponding labels y_j (assumed +/-1 in this formulation),
    optimizing for vector w (the weight vector) and scalar b (the bias),
    with a corresponding linear (affine) model function f(x) = w'\*x - b which
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
            via :py:meth:`SSVMClassifier.fit`, defining the normal vector to the separating hyperplane.

        bias_ (float): Affine shift of the hyperplane obtained by fitting the SSVM classifier via
            :py:meth:`SSVMClassifier.fit`.

        pred_labels_ (list of length n_samples): Prediction labels of test data obtained by predicting with the SSVM
            classifier via :py:meth:`SSVMClassifier.predict`

        classes_ (ndarray of shape (n_classes,)): The class labels.
    '''

    def __init__(self,
                 C: float = 1.0,
                 tol: float = 0.001,
                 solver: object = None,
                 errorTrace: object = None,
                 use_cuda: bool = False,
                 verbosity: int = 0,
                 debug: bool = False):

        # Solver parameters
        self.C = C  # the margin weight
        self.tol = tol  # error tolerance for interior point solver
        self._solver = solver  # solver for solving the LP
        self.errorTrace = errorTrace
        self.use_cuda = use_cuda  # Flag to attempt to use CUDA.
        self.verbosity = verbosity  # Level of verbosity
        self.debug = debug

        # Solver attributes
        self.weights_ = None
        self.bias_ = None
        self.pred_labels_ = None
        self.classes_ = None

    @property
    def solver(self):
        if self._solver is None:
            return LPPrimalDualPy
        else:
            return self._solver

    def fit(self, X, y):
        '''
        Fit/training step for Sparse Support Vector Machines (SSVM). A model function

        f(x)=w'\*x - b

        is found for vector w=len(x) and scalar b which optimally classify the
        training X, in the sense of solving the L1 minimization problem

        min ||w||_1 + C\*sum( xi_j ) s.t. y_j\*(w'\*x_j -b) <= 1-xi_j, j=1,...,n,

        where x_j are vector input X, y_j are {-1,+1} class labels for each x_j,
        xi_j are scalar slack variables.

        This code only supports binary classification right now.

        The weight vector w and bias b are stored in :py:attr:`SSVMClassifier.weights_` and
        :py:attr:`SSVMClassifier.bias_` respectively.
        '''

        import numpy as np
        try:
            import torch
        except ImportError:
            torch = None

        # Check that the stars have aligned so that we can use CUDA.
        use_cuda = self.use_cuda and torch and torch.cuda.is_available()
        if self.verbosity > 0:
            if self.use_cuda and not use_cuda:
                print(
                    'PyTorch could not be imported, or could not access the GPU. Falling back to numpy implementation.')
        #

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        if len(self.classes_) != 2:
            raise ValueError(
                "The supplied training X has fewer or greater than two labels.\nOnly binary classification is supported.")

        # Need an extra step here - SSVM wants labels -1 and 1.
        labelDict = {self.classes_[0]: -1, self.classes_[1]: 1}
        internalLabels = [labelDict[sample] for sample in y]

        nSamples, inputDim = np.shape(X)

        IP = np.diag(np.ones(nSamples)).astype(int)
        eP = np.ones(nSamples).reshape(-1, 1)
        eDim = np.ones(inputDim).reshape(-1, 1)

        D = np.diag(internalLabels)  # Diagonal matrix of labels

        if use_cuda:
            D_c = torch.from_numpy(D).double().cuda()
            trData_c = torch.from_numpy(X).double().cuda()
            DX = torch.mm(D_c, trData_c).cpu().numpy()
            eP_c = torch.from_numpy(eP).double().cuda()
            De = torch.mm(D_c, eP_c).cpu().numpy()
        else:
            DX = np.dot(D, X)
            De = np.dot(D, eP)
        #

        A = np.hstack((DX, -DX, -De, De, IP))
        c = np.vstack((eDim, eDim, np.array([0]).reshape(-1, 1), np.array([0]).reshape(-1, 1), self.C * eP))

        x = self.solver(-c, -A, -eP, output_flag=0, use_cuda=use_cuda, verbosity=self.verbosity, debug=self.debug)

        self.weights_ = x[:inputDim] - x[inputDim:2 * inputDim]
        self.weights_ = self.weights_.reshape(-1, )
        self.bias_ = x[2 * inputDim] - x[2 * inputDim + 1]

        return self

    #
    def predict(self, X, prob=False, pos=None):
        '''
        Classification step for Sparse Support Vector Machine (SSVM).
        After the fit/training step, vectors w and b are found to
        optimally classify the training X (in the sense described
        in the fit() docstring). New X is classified using

        sign(f(x)) = sign( w'\*x - b ).
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
                print(
                    'PyTorch could not be imported, or could not access the GPU. Falling back to numpy implementation.')

        b = self.bias_
        w = self.weights_.reshape(-1, 1)

        if use_cuda:
            data_c = torch.from_numpy(X).double().cuda()
            w_c = torch.from_numpy(w).double().cuda()
            b_c = torch.from_numpy(b).double().cuda()
            d = torch.addmm(-1, b_c, data_c, w_c).cpu().numpy()
        else:
            d = np.dot(X, w) - b

        # use probability or not
        if prob:
            if pos is None:
                pos = self.classes_[0]
            # hard classification past margins
            d[d < -1] = -1
            d[d > 1] = 1
            # shift-scale [-1, 1] to [0, 1]
            d = (d + 1) / 2
            if pos == self.classes_[0]:
                d = 1 - d
            elif pos == self.classes_[1]:
                pass
            else:
                raise ValueError("Positive label provided is not in class labels.")
            pred_labels = d.reshape(-1, ).tolist()
            return pred_labels

        else:
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


class L1SVM(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 nu: float = 1,
                 eps: float = 1e-5,
                 tp: float = .1,
                 delta: float = .001,
                 imax: int = 50,
                 tol: float = 1e-3,
                 kernel_args: dict = None,
                 device: int = -1,
                 verbosity: int = 1):

        # set params
        self.nu = nu
        self.eps = eps
        self.kernel_args = kernel_args
        self.device = device
        self.tp = tp
        self.delta = delta
        self.imax = imax,
        self.tol = tol,
        self.verbosity = verbosity
        self.solver_ = LPNewton(verbosity=verbosity,
                                delta=delta,
                                imax=imax,
                                tol=tol,
                                device=device)

        # set attributes
        self.classes_ = None
        self.label_dict_ = None
        self.w_ = None
        self.gamma_ = None
        self.xi_ = None
        self.X_ = None
        self.y_ = None
        self.sigma_ = None
        self.mu_ = None


    def fit(self, X, y):

        # set randomness
        rng = np.random.default_rng(12345)

        # check that X and y have correct shape
        X, y = check_X_y(X, y)

        # store data as attributes
        self.X_ = X = np.array(X)
        self.y_ = y = np.array(y)

        # mean center and scale each feature to var = 1
        self.mu_ = np.nanmean(X, axis=0).reshape(1, -1)
        self.sigma_ = np.nanstd(X, axis=0).reshape(1, -1)
        X_tr = (X - self.mu_) / self.sigma_


        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        if len(self.classes_) != 2:
            raise ValueError(
                "The supplied training X has fewer or greater than two labels.\nOnly binary classification is supported.")

        # Need an extra step here - SSVM wants labels -1 and 1.
        self.label_dict_ = {self.classes_[0]: -1, self.classes_[1]: 1}
        D = np.array([self.label_dict_[sample] for sample in y]).reshape(1, -1)

        # apply kernel
        if self.kernel_args is not None:
            X_tr = pairwise_kernels(X_tr, **self.kernel_args) * D

        # grab shapes
        m, n = X_tr.shape

        # convert dtypes
        X_tr = self.convert_type(X_tr)
        D = self.convert_type(D.reshape(-1, 1))

        # set params
        nu = self.nu
        eps = self.eps

        # print to console
        if self.verbosity > 0:
            print("Computing the unconstrained objective function f and its derivatives df and d^2f...")

        # Define variables
        en = self.convert_type(np.ones((n, 1)))
        em = self.convert_type(np.ones((m, 1)))
        z = D * X_tr
        w = nu * em
        g = (em * D).t()
        zero = self.convert_type(0)

        # define f, df, hf (func, deriv, hessian resp.)
        f = lambda u: -eps * tc.dot(em.view(-1, ), u.view(-1, )) + (1 / 2) * (tc.pow(tc.norm(tc.relu((tc.matmul(z.t(), u) - en))), 2) + tc.pow(tc.norm(tc.relu((tc.matmul(-z.t(), u) - en))), 2) + tc.pow(-tc.dot(g.view(-1, ), u.view(-1, )), 2) + tc.pow(tc.norm(tc.relu(u - w)), 2) + tc.pow(tc.norm(tc.relu(-u)), 2) )

        df = lambda u: -eps * em + tc.matmul(z, tc.relu(tc.matmul(z.t(), u) - en)) - tc.matmul(z, tc.relu(tc.matmul(-z.t(), u) - en)) + tc.dot(g.view(-1,), u.view(-1,)) * g.t() + tc.relu(u - w) - tc.relu(-u)

        hf = lambda u: tc.matmul(z * (tc.heaviside(tc.abs(tc.matmul(z.t(), u)) - en, zero)).t(), z.t()) + g.t() * g + tc.diag((tc.heaviside(u - w, zero) + tc.heaviside(-u, zero)).view(-1,))

        # solve LP with LPNewton
        if self.verbosity > 0:
            print("Solving with LPNewton:")
        u0 = self.convert_type(rng.normal(size=em.shape))
        u, _ = self.solver_.solve(u0, f, df, hf)

        # update model attributes
        self.w_ = ((1 / eps) * (tc.relu(tc.matmul(z.t(), u) - en) - tc.relu(tc.matmul(-z.t(), u) - en))).detach().cpu().numpy().reshape(-1,)
        self.gamma_ = (- (1 / eps) * tc.dot(g.view(-1,), u.view(-1,))).detach().cpu().numpy()
        self.xi_ = ((1 / eps) * tc.relu(u - w)).detach().cpu().numpy().reshape(-1,)

    def predict(self, X):

        # convert to numpy array
        X = np.array(X)

        # compute D vector
        D = np.array([self.label_dict_[sample] for sample in self.y_]).reshape(-1, 1)

        # center and scale data
        X_tr = (self.X_ - self.mu_) / self.sigma_
        X_tst = (X - self.mu_) / self.sigma_

        # compute decision function
        if self.kernel_args is None:
            p = np.matmul(X_tst, self.w_.reshape(-1, 1)) - self.gamma_


        else:
            p = self.nu * np.matmul(pairwise_kernels(X_tst, X_tr, **self.kernel_args), D * self.w_.reshape(-1, 1)) - self.gamma_


        # compute inverse label dictionary
        inv_label_dict_ = {v: k for k, v in self.label_dict_.items()}

        # predict labels
        y_pred = np.ones_like(p)
        y_pred[p < 0] = -1
        y_pred = np.array([inv_label_dict_[val.item()] for val in y_pred])

        return y_pred

    def convert_type(self, x):

        if self.device == -1:
            if isinstance(x, tc.Tensor):
                return x.detach().cpu().type(tc.float64)
            else:
                return tc.tensor(data=x, dtype=tc.float64)
        elif self.device == 'any':
            if isinstance(x, tc.Tensor):
                return x.detach().type(tc.float64).cuda()
            else:
                return tc.tensor(data=x, dtype=tc.float64).cuda()
        else:
            cuda = tc.device('cuda:' + str(self.device))
            if isinstance(x, tc.Tensor):
                return x.detach().type(tc.float64).to(cuda)
            else:
                return tc.tensor(data=x, device=cuda, dtype=tc.float64)


class SSVMSelect(SSVMClassifier):

    def __init__(self,
                 C: float = 1.0,
                 tol: float = 0.001,
                 solver: object = None,
                 errorTrace: object = None,
                 use_cuda: bool = False,
                 verbosity: int = 0,
                 debug: bool = False,
                 jump_ratio: float = 5.0,
                 n_features: int = None,
                 show_plot: bool = True,
                 corr_threshold: float = None,
                 ):

        super(SSVMSelect, self).__init__(C=C,
                                         tol=tol,
                                         solver=solver,
                                         errorTrace=errorTrace,
                                         use_cuda=use_cuda,
                                         verbosity=verbosity,
                                         debug=debug,
                                         )

        # set attributes
        self.jump_ratio = jump_ratio
        self.n_features = n_features
        self.show_plot = show_plot
        self.corr_threshold = corr_threshold
        self.features = None
        self.correlates = None
        self.f_ranks = None

    def fit(self, X, y):

        # call super
        super(SSVMSelect, self).fit(X, y)

        # select the features
        self.features = self.select_features(X)

        # pull feature weights
        f_weights = np.abs(self.weights_)

        # sort the weights
        S = np.argsort(-f_weights)
        f_weights_sorted = f_weights[S]

        # compute ranks
        self.f_ranks = pd.DataFrame(index=np.arange(len(f_weights)), data=S.reshape(-1, 1), columns=[["Ranks"]])

        if self.show_plot:
            self.plot_weights()

        return self

    def select_features(self, X):

        # pull feature weights
        f_weights = np.abs(self.weights_)

        # sort the weights
        S = np.argsort(-f_weights)
        f_weights_sorted = f_weights[S]

        if self.n_features is None:
            a = f_weights_sorted[:-1]
            b = f_weights_sorted[1:]
            f_ratios = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
            try:
                id = np.where(f_ratios > self.jump_ratio)[0][0] + 1
                print("%d features selected!" % (id, ))
                features = S[:id]
                self.n_features = id
            except IndexError:
                print("Jump failed, no features selected, resorting to number of features provided by user...")
                assert self.n_features is not None, "User did not provide the number of top features and the jump failed. Aborting feature selection!"
                features = S[:self.n_features]
        else:
            features = S[:self.n_features]

        if self.corr_threshold is not None:
            print("Generating correlated features using threshold: %0.2f..." % (self.corr_threshold,))
            self.correlates = get_correlates(features, X, self.corr_threshold)
            feaures = np.concatenate(features, self.correlates, axis=None)

        return features

    def plot_weights(self):
        # pull feature weights
        f_weights = np.abs(self.weights_)

        # sort the weights
        S = np.argsort(-f_weights)
        f_weights_sorted = f_weights[S]

        # plot the weights
        fig = plt.figure(figsize=(15, 10))
        ax = plt.axes()
        ax.semilogy(f_weights_sorted)
        #ax.scatter(np.arange(len(f_weights)), f_weights_sorted)
        #ax.set_yscale('symlog')

        # set axis labels and titles
        ax.set(xlabel='Features',
               ylabel='|Weight|',
               title='SSVM Weights')

        # set font sizes
        ax.set_title(ax.get_title(), fontsize=16)
        ax.set_xlabel(ax.get_xlabel(), fontsize=18)
        ax.set_ylabel(ax.get_ylabel(), fontsize=18)

        # plot
        plt.show()

        return fig, ax

    def transform(self, X, y=None):

        # check for fit
        check_is_fitted(self)

        # check array
        X = np.array(X)

        # restrict data
        return X[:, self.features]