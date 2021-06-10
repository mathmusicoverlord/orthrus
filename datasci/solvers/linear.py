"""
This module contain solvers for linear programs.
"""

import torch as tc
from copy import deepcopy

class LPNewton():
    def __init__(self,
                delta: float = .001,
                device: int = -1,
                tol: float = 1e-3,
                imax: int = 50,
                verbosity: int = 1):

        self.delta = delta
        self.tol = tol
        self.imax = imax
        self.device = device
        self.verbosity = verbosity

    def solve(self, u0, f, df, hf):

        # convert dtype
        ui = self.convert_type(deepcopy(u0))

        # grab shape
        n = u0.shape[0]

        # create delta*I matrix
        dI = self.delta*self.convert_type(tc.eye(n))

        # set default exit status
        exit_status = 1

        # compute u
        for i in range(self.imax):
            # perform LPNewton step
            if self.verbosity > 0:
                print("\tPerforming LPNewton step %d..." % (i,))
            uii = self.step(ui, f, df, hf, dI)

            # check for convergence
            err = tc.norm(ui - uii).item()
            if self.verbosity > 0:
                print("\t||u_i - u_{i+1}|| = %0.2f" % (err,))
            if err < self.tol:
                ui = uii
                exit_status = 0
                break

            # update ui
            ui = uii

        # print details
        if self.verbosity > 0:
            if exit_status == 0:
                print("\tSolution converged!")
            elif exit_status == 1:
                print("\tMaximum iterations reached!")

        # return the solution
        return ui, exit_status


    def step(self, ui, f, df, hf, dI):

        # compute derivative and hessian at ui
        df_ui = df(ui)
        hf_ui = hf(ui)

        # compute di
        di, _ = tc.solve(-df_ui, hf_ui + dI)

        # determine Armijo step size
        if self.verbosity > 0:
            print("\tDetermining Armijo step size...")
        l = 1
        while (f(ui) - f(ui + l * di)) < -(l / 4)*tc.matmul(df_ui.t(), di):
            l = (1 / 2)*l

        print("\tObjective = %0.2f" % (f(ui),))

        # return the updated ui
        return ui + (l * di)

    def convert_type(self, x):

        if self.device == -1:
            if isinstance(x, tc.Tensor):
                return x.detach().cpu().type(tc.float64)
            else:
                return tc.tensor(data=x, dtype=tc.float64)
        else:
            cuda = tc.device('cuda:' + str(self.device))
            if isinstance(x, tc.Tensor):
                return x.detach().type(tc.float64).to(cuda)
            else:
                return tc.tensor(data=x, device=cuda, dtype=tc.float64)