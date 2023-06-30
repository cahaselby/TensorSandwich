
"""
27 MAR 2023

This is the module needed to run tensor sandwich trial

author: Cullen Haselby 
"""
#########################
# IMPORTS
#########################

import numpy as np
from scipy.linalg import khatri_rao,hilbert,null_space,subspace_angles, hadamard, qr
import matplotlib.pyplot as plt
import tensorly as tl
from tensorly.decomposition import parafac, tucker
import cvxpy
from solver import Solver
from sklearn.utils import check_array
import timeit

#########################
# Helper functions
#########################

def add_noise(T,relnoise):

    Noise = np.random.normal(size=T.shape)
    #Normalize 
    Noise = Noise / tl.norm(Noise)
    #Scale relative to norm of T
    Noise = (relnoise *tl.norm(T) )* Noise

    return T +  Noise

def rel_error(approxT,exactT):
    return tl.norm(approxT - exactT) / tl.norm(exactT)

def censored_lstsq(A, B, M):
    """Solves least squares problem subject to missing data.

    Note: uses a broadcasted solve for speed.

    Args
    ----
    A (ndarray) : m x r matrix
    B (ndarray) : m x n matrix
    M (ndarray) : m x n binary matrix (zeros indicate missing values)

    Returns
    -------
    X (ndarray) : r x n matrix that minimizes norm(M*(AX - B))
    """

    # Note: we should check A is full rank but we won't bother...

    # if B is a vector, simply drop out corresponding rows in A
    if B.ndim == 1 or B.shape[1] == 1:
        return np.linalg.lstsq(A[M], B[M],rcond=None)[0]

    # else solve via tensor representation
    rhs = np.dot(A.T, M * B).T[:,:,None] # n x r x 1 tensor
    T = np.matmul(A.T[None,:,:], M.T[:,:,None] * A[None,:,:]) # n x r x r tensor
    try:
        output = np.squeeze(np.linalg.solve(T, rhs)).T # transpose to get r x n
    except np.linalg.LinAlgError:
        print("system is probably singular, switching to least square mode")
        
        output = np.empty((A.shape[1], B.shape[1]))
        for i in range(B.shape[1]):
            m = M[:,i] # drop rows where mask is zero
            output[:,i] = np.linalg.lstsq(A[m], B[m,i])[0]
            
    return output


def twophase_mat_sampler(M,r,beta,budget): 
    (n,m) = M.shape
    Omega_1 = round(beta*budget)

    mask = np.zeros(n*m, dtype="bool")
    mask[:Omega_1] = True

    np.random.shuffle(mask)
    mask = mask.reshape((n,m))
    P_Omega = np.zeros_like(M)
    P_Omega[mask] = M[mask]
    U,S,Vt = np.linalg.svd(P_Omega,full_matrices=False)
    mu_A = (n/r)*np.linalg.norm(U[:,:r],axis=1)**2
    nu_A = (m/r)*np.linalg.norm(Vt[:r,:],axis=0)**2


    weights = []
    idx_choice = []
    for i in range (n):
        for j in range(m):
            if mask[i,j] == 1:
                weights.append(0)
            else:
                weights.append((r/n)*(mu_A[i]+ nu_A[j])*(np.log(2*n)**2))

    weights = weights / np.sum(weights)

    draw = np.random.choice(int(n**2), size=round((1-beta)*budget),p=weights, replace=False)
    idxs = []
    for d in draw:
        i = d // n
        j = d % n
        idxs.append((i,j))
        mask[i,j] = True
    
    return mask

#########################
# Monkey Patch of the NucNorm Solver class from fancy impute, the only change is to switch out the solver
#########################
class NuclearNormMinimization(Solver):
    """
    Simple implementation of "Exact Matrix Completion via Convex Optimization"
    by Emmanuel Candes and Benjamin Recht using cvxpy.
    """

    def __init__(
            self,
            require_symmetric_solution=False,
            min_value=None,
            max_value=None,
            error_tolerance=1e-8,
            max_iters=2500,
            verbose=True):
        """
        Parameters
        ----------
        require_symmetric_solution : bool
            Add symmetry constraint to convex problem
        min_value : float
            Smallest possible imputed value
        max_value : float
            Largest possible imputed value
        error_tolerance : bool
            Degree of error allowed on reconstructed values. If omitted then
            defaults to 0.0001
        max_iters : int
            Maximum number of iterations for the convex solver
        verbose : bool
            Print debug info
        """
        Solver.__init__(
            self,
            min_value=min_value,
            max_value=max_value)
        self.require_symmetric_solution = require_symmetric_solution
        self.error_tolerance = error_tolerance
        self.max_iters = max_iters
        self.verbose = verbose

    def _constraints(self, X, missing_mask, S, error_tolerance):
        """
        Parameters
        ----------
        X : np.array
            Data matrix with missing values filled in
        missing_mask : np.array
            Boolean array indicating where missing values were
        S : cvxpy.Variable
            Representation of solution variable
        """
        ok_mask = ~missing_mask
        masked_X = cvxpy.multiply(ok_mask, X)
        masked_S = cvxpy.multiply(ok_mask, S)
        abs_diff = cvxpy.abs(masked_S - masked_X)
        close_to_data = abs_diff <= error_tolerance
        constraints = [close_to_data]
        if self.require_symmetric_solution:
            constraints.append(S == S.T)

        if self.min_value is not None:
            constraints.append(S >= self.min_value)

        if self.max_value is not None:
            constraints.append(S <= self.max_value)

        return constraints

    def _create_objective(self, m, n):
        """
        Parameters
        ----------
        m, n : int
            Dimensions that of solution matrix
        Returns the objective function and a variable representing the
        solution to the convex optimization problem.
        """
        # S is the completed matrix
        shape = (m, n)
        S = cvxpy.Variable(shape, name="S")
        norm = cvxpy.norm(S, "nuc")
        objective = cvxpy.Minimize(norm)
        return S, objective

    def solve(self, X, missing_mask):
        X = check_array(X, force_all_finite=False)

        m, n = X.shape
        S, objective = self._create_objective(m, n)
        constraints = self._constraints(
            X=X,
            missing_mask=missing_mask,
            S=S,
            error_tolerance=self.error_tolerance)
        problem = cvxpy.Problem(objective, constraints)
        problem.solve(
            verbose=True,
            solver=cvxpy.SCS,
            max_iters=self.max_iters,
            eps = self.error_tolerance,
            # use_indirect, see: https://github.com/cvxgrp/cvxpy/issues/547
            use_indirect=False)
        return S.value
    
def tensor_sandwich(T_true,T_noise,N,r,relnoise,n_slices,alpha,beta,delta,gamma,full_output=False):

    slice_idx = np.random.choice(N,n_slices,replace=False)
    
    budget = min(round(gamma*N*r*np.log(N)**2),round(N**2))


    
    Mask = tl.zeros(T_true.shape,dtype=bool)
    
    completed_slices = tl.zeros(Mask[:,:,:n_slices].shape)
    for i,s in enumerate(slice_idx):
        Mask[:,:,s] = twophase_mat_sampler(T_true[:,:,s],r,beta,budget)
        completed_slices[:,:,i] = NuclearNormMinimization().solve(T_noise[:,:,s],~Mask[:,:,s] )
    T_tilde_3 = tl.unfold(T_noise,2).T
    M_3 = tl.unfold(Mask,2).T    
    
    #random vectors for projecting
    ga = np.random.normal(size=n_slices)
    gb = np.random.normal(size=n_slices)
    ga = ga / np.linalg.norm(ga)
    gb = gb / np.linalg.norm(ga)
    #contract along mode three using the random vector
    Ta = (completed_slices @ ga).reshape(N,N)
    Tb = (completed_slices @ gb).reshape(N,N) 
    #Now threshold the SVD of Ta and Tb to the given rank
    Ua, Sa, VaT = np.linalg.svd(Ta, full_matrices=0)
    Ta = Ua[:,:r]@ np.diag(Sa[:r]) @ VaT[:r,:]
    Ub, Sb, VbT = np.linalg.svd(Tb, full_matrices=0)
    Tb = Ub[:,:r]@ np.diag(Sb[:r]) @ VbT[:r,:]
    #Using the regularized rank r Ta and Tb, compute X and its eigen decomposition
    X = (np.linalg.pinv(Ta) @ Tb).T
    evalsV, evecsV = np.linalg.eig(X)
    #Find the indices which match the largest eigenvalues by magnitude and sort in the in descending order
    evalsV_idx = np.argsort(-np.abs(evalsV))
    #Pick out the eigenvectors that match
    #LearnedV = evecsV[:,evalsV_idx[:r]]
    LearnedV = evecsV[:,:r]
    #Invert V and use its pseudoinverse to find the matching U
    LearnedU = Ta @ np.linalg.pinv(LearnedV.T)
    #These U may not be normalized (?)
    #LearnedU = LearnedU / np.linalg.norm(LearnedU, axis=0)
    #Setup the linear system and solve for the third factor vector
    A = khatri_rao(LearnedU, LearnedV)
    Q,R,P = qr(A.T, pivoting=True)
    oversample = round(delta*r)
    M_3[P[:oversample]] = 1
    res = np.linalg.lstsq(A, T_tilde_3, rcond=None)
    LearnedW = censored_lstsq(A, T_tilde_3, M_3).T
    T_recon = tl.cp_to_tensor((np.ones(r), [LearnedU,LearnedV,LearnedW]))
    error = rel_error(T_recon,T_true)
    
    if full_output: return [error,budget,np.sum(M_3),np.sum(M_3)/N**3],(np.ones(r), [LearnedU,LearnedV,LearnedW]), tl.fold(M_3,2,(N,N,N))
    else: return [error,budget,np.sum(M_3),np.sum(M_3)/N**3]

def tensor_als(T_true,T_noise,N,r,relnoise,budget,alpha,max_iter,init='svd',mask=None):

    if mask is None:
        sample_mask = np.zeros(N**3, dtype="bool")
        sample_mask[:int(budget)] = True
        np.random.shuffle(sample_mask)

        sample_mask = sample_mask.reshape((N,N,N))
    else: 
        sample_mask = mask
        
    res = parafac(T_noise,rank=r, n_iter_max=max_iter,init=init,mask=sample_mask)
    T_recon = tl.cp_to_tensor(res)
    error = rel_error(T_recon,T_true)
    
    return [error]


def tucker_core_to_cp(TuckerSpaces, CoreCPSpaces):
    """Convenience function which takes the rxr CPD factors obtained from a (rxrxr) core tensor from a Tucker decomp, along with the Nxr Tucker factor matrices and
    're-inflates' these to the full NxNxN tensor with a rank r CPD. This is a trick Haselby observed for speeding up the Tucker to CPD step of the overall algorithm
    
    ----
    TuckerSpaces (list) : List of Nxr matrices that are the factor matrices in a Tucker decomp
    CoreCPSpaces (list) : list of rxr matrices that are the factor matrices of a CPD of the core tensor of a Tucker decomp
    -------
    K (array) : NxNxN tensor with CPD of rank r
    """
    N = TuckerSpaces[0].shape[0]
    r = TuckerSpaces[0].shape[1]

    K = khatri_rao(TuckerSpaces[0] @ CoreCPSpaces[0], TuckerSpaces[1] @ CoreCPSpaces[1])
    K = khatri_rao(K, TuckerSpaces[2] @ CoreCPSpaces[2])
    K = K@np.ones(r)
    return K.reshape(N,N,N)
  
def core_censored_solve(T,F,sample_mask,N,r):
    """Solves censored least squares problem to find the core of a Tucker, given factors and a tensor with missing values according to sample mask.
    ----
    T (array) : 3-way tensor assumed to be cube of size NxNxN that has missing data as described by the given mask
    F (list) : List of Nxr arrays that are estimates for the subspaces spanned by the true factor matrices of the tensor
    sample_mask (array) : 1D boolean array with N^3 entries. when reshaped, should be the sample pattern for the missing values
    N (int) : dimension of any one of the modes of the tensor
    r (int) : rank
    -------
    Tucker_T (array) : Tensor NxNxN with tucker decomposition with factors F and best fit core according to censored least squares

    """
    Projector = tl.tenalg.kronecker(F)
    Core = censored_lstsq(Projector, T.reshape(-1), sample_mask)
    return (Projector@Core).reshape(N,N,N), Core.reshape(r,r,r)

def initilization(T,sample_mask,N,p,r,c,mu,sigma_1,sigma_r):
    """ Given a tensor where values are missing per the sample mask find an estimate for the subspaces spanned by its factor matrices
    ----
    T (array) : 3-way tensor assumed to be cube of size NxNxN
    sample_mask : 1D boolean array with N^3 entries. when reshaped, should be the sample pattern for the missing values
    N (int) : dimension of any one of the modes of the tensor
    p (float) :  sample rate of the tensor
    r (int) : assumed rank of the tensor, user must decide
    c (float) : c*sqrt(r/N) bounds the row norm of all factor matrices
    mu (float) :  coherence bound for all the factor matrices
    sigma_1 (float) : largest singular value of any factor matrix
    sigma_r (float) : rank-th smallest singular value. In exact setting would be the smallest nonzero singular value if the tensor is truly rank r.
    -------
    F (list) : List of Nxr arrays that are estimates for the subspaces spanned by the true factor matrices of the tensor

    This algorithm is based on Montarri and Sun, reused by Moitra
    """
    #In this intialization, missing values are zeroed 
    T0 = T.reshape(-1).copy()
    T0[~sample_mask] = 0
    T0 = T0.reshape(N,N,N)

    #number of modes
    M = len(T0.shape)
    F = []
    #tau parameter is used to control the coherence of the estimated factor matrices by zeroing out bad rows
    tau = np.sqrt(r/N) * ((2*mu*r*sigma_1**2) / ((c**2)*(sigma_r**2)))**5
    for m in range(M):
        U = tl.unfold(T0,m)
        D = np.diag(np.diag(U @ U.T))
        B = (1/p)*D + (1/p**2)*(U @ U.T - D)

        #Zero out rows in the factor matrix which have a norm that is larger than the tau number calculated
        Ur, _, _ = np.linalg.svd(B)
        X = Ur[:,:r]
        r_norms = np.linalg.norm(X, axis=1)
        X[r_norms > tau] = 0
        
        #Just need a set of orthonormal basis elements for the space spanned bu X, Q from QR will do the trick
        Qr, _= np.linalg.qr(X)

        F.append(Qr)

    return F

def kron_alt_min(T,F,N,r,sample_mask,p,sigma_1,sigma_r,k=10):
    """ Given a tensor where values are missing per the sample mask and an estimate of the factor matrices find a (better) estimate of the subspaces spanned by its factor matrices
    ----
    T (array) : 3-way tensor assumed to be cube of size NxNxN
    F (list) : List of three factor matrices for each of the modes of size Nxr
    sample_mask (array) : 1D boolean array with N^3 entries. when reshaped, should be the sample pattern for the missing values
    N (int) : dimension of any one of the modes of the tensor
    r (int) : assumed rank of the tensor, user must decide
    p (float) :  sample rate of the tensor
    sigma_1 (float) : largest singular value of any factor matrix
    sigma_r (float) : rank-th smallest singular value. In exact setting would be the smallest nonzero singular value if the tensor is truly rank r.
    sub_sample (float) : default is 0.5. Subsample rate is used to vary which samples are considered in each run of outer loop. Helps performance, and may help avoid getting stuck in local mins
    k (int) : number of iterations to run
    -------
    F (list) : List of Nxr arrays that are estimates for the subspaces spanned by the true factor matrices of the tensor

    This algorithm is described in Moitra, Liu. It is essentially a Tucker decomposition that uses censored least square in the inner loop
    """

    #Here is the theoritcal number of iterations you'd need to run
    #to get the exact results stated in paper. It is way too large to use in practice I have discovered
    #k = int( 100*np.log((N*sigma_1) / (c*sigma_r) ) )
    
    #subsample rate for their proofs, also found this to be bad in practice
    #p_prime = p / k

    for t in range(k):
        V = F
        M = sample_mask.reshape((N,N,N))
        for m in range(3):
            modes = list(range(3))
            modes.remove(m)
            L=modes[0]
            R=modes[1]
            Mm = tl.unfold(M,m)
            B = np.linalg.qr(np.kron(V[L],V[R]))[0].T

            H = censored_lstsq(B.T,tl.unfold(T,m).T,Mm.T)
            Ur, _, _ = np.linalg.svd(H.T)
            F[m] = Ur[:,:r]

    return F

def iwen_jennrich(T_tilde,N,r):

    T_tilde_3 = tl.unfold(T_tilde,2).T
    
    #random vectors for projecting
    ga = np.random.normal(size=N)
    gb = np.random.normal(size=N)
    ga = ga / np.linalg.norm(ga)
    gb = gb / np.linalg.norm(ga)
       
    #contract along mode three using the random vector
    Ta = (T_tilde_3 @ ga).reshape(N,N)
    Tb = (T_tilde_3 @ gb).reshape(N,N) 
    
    #Now threshold the SVD of Ta and Tb to the given rank
    Ua, Sa, VaT = np.linalg.svd(Ta, full_matrices=0)
    Ta = Ua[:,:r]@ np.diag(Sa[:r]) @ VaT[:r,:]

    Ub, Sb, VbT = np.linalg.svd(Tb, full_matrices=0)
    Tb = Ub[:,:r]@ np.diag(Sb[:r]) @ VbT[:r,:]

    #Using the regularized rank r Ta and Tb, compute X and its eigen decomposition
    X = (np.linalg.pinv(Ta) @ Tb).T
    evalsV, evecsV = np.linalg.eig(X)
    
    #Find the indices which match the largest eigenvalues by magnitude and sort in the in descending order
    evalsV_idx = np.argsort(-np.abs(evalsV))
    #Pick out the eigenvectors that match
    LearnedV = evecsV[:,evalsV_idx[:r]]

    #Invert V and use its pseudoinverse to find the matching U
    LearnedU = Ta @ np.linalg.pinv(LearnedV.T)

    #These U may not be normalized (?)
    #LearnedU = LearnedU / np.linalg.norm(LearnedU, axis=0)
    
    #Setup the linear system and solve for the third factor vector
    A = khatri_rao(LearnedU, LearnedV)
    res = np.linalg.lstsq(A, T_tilde_3, rcond=None)
    LearnedW = res[0].T

    IwenFactors = [LearnedU, LearnedV, LearnedW]
    
    return IwenFactors

def org_jennrich(T_tilde,N,r):
    """ Given a tensor finds the CPD of a given rank based on Jennrich's algorithm
    ----
    T_tilde (array) : 3-way tensor assumed to be cube of size NxNxN. 

    N (int) : dimension of any one of the modes of the tensor
    r (int) : assumed rank of the tensor, user must decide
    -------
    JenFactors (list) : List of Nxr arrays that are matrices of the tensor for a CPD of rank r. In the exact setting and with assumptions about the linear independence of components, would be exact

    This algorithm is described orginally by Harshman, and appeared many times in the literature since
    """

    T_tilde_3 = tl.unfold(T_tilde,2).T
 
    #random vectors for projecting
    ga = np.random.normal(size=N)
    gb = np.random.normal(size=N)
    ga = ga / np.linalg.norm(ga)
    gb = gb / np.linalg.norm(ga)
       
    #contract along mode three using the random vector
    Ta = (T_tilde_3 @ ga).reshape(N,N)
    Tb = (T_tilde_3 @ gb).reshape(N,N) 
    

    X = Ta @ np.linalg.pinv(Tb)
    Y = (np.linalg.pinv(Ta) @ Tb).T

    #compute the eigen decompositions of X any Y    
    evalsU, evecsU = np.linalg.eig(X)
    evalsV, evecsV = np.linalg.eig(Y)
    
    #Find the sorting of the eigenvalues by magnitude
    evalsU_idx = np.argsort(np.abs(evalsU))
    evalsV_idx = np.argsort(np.abs(evalsV))
    
    #Truncate to the correct rank, and flip one index so reciprocals will match
    matchedV_idx = evalsV_idx[N-r:]
    if (N-r)==0:
        matchedU_idx = np.flip(evalsU_idx)
    else:
         matchedU_idx = evalsU_idx[-1:(N-r-1):-1]
            
    #Pick out the corresponding eigenvectors
    LearnedU = evecsU[:,matchedU_idx]
    LearnedV = evecsV[:,matchedV_idx]

    #Setup the linear system and solve for the third factor vector
    A = khatri_rao(LearnedU, LearnedV)
    res = np.linalg.lstsq(A, T_tilde_3, rcond=None)
    LearnedW = res[0].T

    JenFactors = [LearnedU, LearnedV, LearnedW]
    return JenFactors