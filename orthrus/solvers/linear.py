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
            if self.verbosity > 1:
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

        if self.verbosity > 1:
            print("\tObjective = %0.2f" % (f(ui),))

        # return the updated ui
        return ui + (l * di)

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



# copied from calcom!
def LPPrimalDualPy(c, A, b, **kwargs):
    '''
    This is a Python implementation of a code solving
    a linear program of the form:

    max c'x
        s.t. Ax>=b, x>=0.

    Note this may be a different ``standard form" than seen elsewhere.
    Original version of this in calcom was named ``rKKTxy" and lived in
    ssvmclassifier.py. Before that, it was a Matlab code by
    Sofya Chepushtanova and Michael Kirby, and later implemented in Python by
    Tomojit Ghosh.

    Inputs:

        c, np.ndarray of type float or double, with c.shape = (N,1)
        A, np.ndarray of type float or double, with A.shape = (M,N)
        b, np.ndarray of type float or double, with b.shape = (M,1)

    Optional inputs:

        output_flag     (integer, default 0, indicating form of the output)
        verbosity       (integer, default 0. Nonzero => print statements used)
        use_cuda        (boolean, default False. indicating whether to use CUDA or not)

        max_iters       (integer, default 200)
        delta           (float or double, default 0.1)
        tol             (float or double, default 10**-3)
        dtheta_min      (float or double, default 10**-9)

        debug           (boolean, default False. If True, immediately turns on the Python debugger.)

    Outputs: depends on the value of output_flag:

        output_flag==0: (default)
            The minimizer x, an np.ndarray with x.shape = (N,)
        output_flag==1:
            IP, a dictionary which has detailed information about the
            solver and solution. This is used by
            calcom.classifiers.SSVMClassifier(), as it has auxilliary
            information used in solving the sparse SVM optimization
            problem specifically.

            NOTE: this may change in the future to make this code less beholden
                to SSVM.

    '''
    if kwargs.get('debug', False):
        import pdb
        pdb.set_trace()
    #

    import numpy as np
    import torch

    max_iters = kwargs.get('max_iters', 200)
    delta = kwargs.get('delta', 0.1)
    tol = kwargs.get('tol', 0.001)

    dtheta_min = kwargs.get('dtheta_min', np.double(10.) ** -9)
    output_flag = kwargs.get('output_flag', 0)
    verbosity = kwargs.get('verbosity', 0)

    use_cuda = kwargs.get('use_cuda', False)  # set to True for large enough problem sizes

    N = len(c)
    M = len(b)

    # TODO: (check)
    # Is there a reason these are initialized in this way instead of randn?
    x, z, e3 = np.ones(N).reshape(-1, 1), np.ones(N).reshape(-1, 1), np.ones(N).reshape(-1, 1)
    p, w, e2 = np.ones(M).reshape(-1, 1), np.ones(M).reshape(-1, 1), np.ones(M).reshape(-1, 1)
    E0 = np.vstack((x, w, p, z))  # initial point
    theta = np.zeros(max_iters)

    err = 999  # Placeholder initial error.

    A_t = torch.from_numpy(A).double()
    b_t = torch.from_numpy(b).double()
    c_t = torch.from_numpy(c).double()
    e2_t = torch.from_numpy(e2).double()
    e3_t = torch.from_numpy(e3).double()
    x_t = torch.from_numpy(x).double()
    p_t = torch.from_numpy(p).double()
    w_t = torch.from_numpy(w).double()
    z_t = torch.from_numpy(z).double()
    theta_t = torch.from_numpy(theta).double()

    # Copy data into GPU
    if use_cuda:
        A_t = A_t.cuda()
        b_t = b_t.cuda()
        c_t = c_t.cuda()
        e2_t = e2_t.cuda()
        e3_t = e3_t.cuda()
        x_t = x_t.cuda()
        p_t = p_t.cuda()
        w_t = w_t.cuda()
        z_t = z_t.cuda()
        theta_t = theta_t.cuda()

    if output_flag == 1:
        IP = {'xx': [], 'ww': [], 'pp': [], 'zz': [], 'met1': [], 'met2': [], 'met3': [], 'met4': [], 'amet1': [],
              'amet2': [], 'amet3': [], 'amet4': [], 'bmet1': [], 'bmet2': [], 'bmet3': [], 'bmet4': [], 'err': err,
              'exitflag': False, 'itrs': 0, 'val': -999}

    for i in range(max_iters):

        if (verbosity):  # May want a second verbosity level for this.
            print('Iteration:', i + 1)
        b1_t = torch.addmm(b_t, A_t, x_t, alpha=-1)
        b2_t = torch.addmm(c_t, A_t.t(), p_t, alpha=-1)
        rho_t = torch.add(b1_t, -1, w_t)
        sig_t = torch.add(b2_t, z_t)
        gamma = (torch.mm(z_t.t(), x_t) + torch.mm(p_t.t(), w_t))[0]
        mu = (delta * gamma) / (M + N)
        a1_t = x_t * z_t
        a2_t = p_t * w_t
        F = torch.cat((rho_t, sig_t, a1_t - (mu * e3_t), a2_t - (mu * e2_t)))
        a3_t = p_t.reciprocal()
        a4_t = x_t.reciprocal()
        a5_t = z_t.reciprocal()
        nn1_t = (x_t * a5_t)
        t1_t = b2_t + (mu * a4_t)
        WW_t = nn1_t.t().repeat(len(A), 1)
        AXZI2_t = A_t * WW_t
        r_t = b1_t - (mu * a3_t) - torch.mm(A_t, (nn1_t * t1_t))
        # this line is causing the most significant difference
        # in the outputs of numpy and pytorch versions
        R_t = - torch.diag((a3_t * w_t)[:, 0]) - torch.mm(AXZI2_t, A_t.t())
        # import pdb;pdb.set_trace()
        # This will throw an exception if there are linear dependencies in the data
        dp_t, _ = torch.solve(r_t.view(-1, 1), R_t)
        #
        dx_t = ((x_t * a5_t).view(-1) * (t1_t - torch.mm(A_t.t(), dp_t)).view(-1)).view(-1, 1)

        dz_t = (mu * a4_t) - z_t - (a4_t * z_t) * dx_t
        dw_t = (mu * a3_t) - w_t - (a3_t * w_t) * dp_t
        mm = torch.max(torch.cat((- dx_t / x_t, - dp_t / p_t, -dz_t / z_t, -dw_t / w_t)))
        newtheta = 0.9 / mm
        theta_t[i] = newtheta
        if theta_t[i] < dtheta_min:  # if the step size is too small stop
            if (verbosity):
                print('Step size is too small: ', theta_t[i])
            #
            break
        #

        x_t = x_t + theta_t[i] * dx_t
        p_t = p_t + theta_t[i] * dp_t
        z_t = z_t + theta_t[i] * dz_t
        w_t = w_t + theta_t[i] * dw_t

        met1 = torch.norm(rho_t)
        met2 = torch.norm(sig_t)
        met3 = gamma
        met4 = torch.norm(F)

        err = np.max([met1.item(), met2.item(), met3.item(), met4.item()])

        if output_flag == 1:
            amet1 = torch.norm(rho_t, 1)
            amet2 = torch.norm(sig_t, 1)
            amet3 = met3
            amet4 = torch.norm(F, 1)
            bmet1 = torch.norm(rho_t, np.inf)
            bmet2 = torch.norm(sig_t, np.inf)
            bmet3 = met3
            bmet4 = torch.norm(F, np.inf)
            x_temp = torch.clone(x_t)
            p_temp = torch.clone(p_t.cpu())
            z_temp = torch.clone(z_t.cpu())
            w_temp = torch.clone(w_t.cpu())

            if use_cuda:
                x_temp = x_temp.cpu()
                p_temp = p_temp.cpu()
                z_temp = z_temp.cpu()
                w_temp = w_temp.cpu()
                met1 = met1.cpu()
                met2 = met2.cpu()
                met3 = met3.cpu()
                met4 = met4.cpu()
                amet1 = amet1.cpu()
                amet2 = amet2.cpu()
                amet3 = met3
                amet4 = amet4.cpu()
                bmet1 = bmet1.cpu()
                bmet2 = bmet2.cpu()
                bmet3 = met3
                bmet4 = bmet4.cpu()

            IP['xx'].append(x_temp.numpy())
            IP['pp'].append(p_temp.numpy())
            IP['zz'].append(z_temp.numpy())
            IP['ww'].append(w_temp.numpy())
            IP['met1'].append(met1.numpy())  # measure of primal constraint
            IP['met2'].append(met2.numpy())  # measure of dual constraint
            IP['met3'].append(met3.numpy())  # measure of complementarity
            IP['met4'].append(met4.numpy())  # value of F that should be zero for a solution
            IP['amet1'].append(amet1)  # measure of primal constraint
            IP['amet2'].append(
                amet2)  # measure of dual constraint    IP['amet3'].append(np.linalg.norm(gamma,1))#measure of complementarity
            IP['amet3'].append(amet3)  # measure of complementarity
            IP['amet4'].append(amet4)  # value of F that should be zero for a solution
            IP['bmet1'].append(bmet1)  # measure of primal constraint
            IP['bmet2'].append(
                bmet2)  # measure of dual constraintIP['bmet3'].append(np.linalg.norm(gamma,np.inf))#measure of complementarity
            IP['bmet3'].append(bmet3)  # measure of complementarity
            IP['bmet4'].append(bmet4)  # value of F that should be zero for a solution

        if (verbosity):  # May want a second verbosity level for this.
            print('Error:', err)
        #

        if output_flag == 0:
            if err < tol:
                break
        else:
            if err < tol:
                val = torch.dot(c_t.view(-1).T, x_t.view(-1))
                if use_cuda:
                    val = val.cpu()
                IP['val'] = val.numpy()
                IP['itrs'] = i
                IP['exitflag'] = True
                IP['err'] = err
                break
            else:
                IP['itrs'] = i
                IP['err'] = err
        #

    # Copy necessary data from GPU to host memory
    if use_cuda:
        x_t = x_t.cpu()
        w_t = w_t.cpu()
        p_t = p_t.cpu()
        z_t = z_t.cpu()

    x = x_t.numpy()
    w = w_t.numpy()
    p = p_t.numpy()
    z = z_t.numpy()

    if output_flag == 1:
        IP['x'] = x
        IP['w'] = w
        IP['p'] = p
        IP['z'] = z
        IP['conerr'] = np.dot(A, x) - b
        return IP
    else:
        return x
    #
#