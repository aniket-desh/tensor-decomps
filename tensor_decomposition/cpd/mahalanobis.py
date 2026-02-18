import numpy as np
import numpy.linalg as la
from .common_kernels import solve_sys, compute_lin_sysN, normalise, cp_reconstruct, mahalanobis_norm
try:
    import Queue as queue
except ImportError:
    import queue


class CP_AMDM_MLE_Optimizer():
    """
    model-matched amdm optimizer for cp decomposition with known covariance structure.
    
    unlike the standard CP_AMDM_Optimizer which estimates the metric from the current
    factors, this optimizer uses the pre-computed metric factors (M^{-1} = M_1^{-1} ⊗ ... ⊗ M_N^{-1})
    from the noise model to perform true mahalanobis-weighted least squares.
    
    this is the correct optimizer for experiment a (mle story) where:
    - we know the noise covariance structure from the generative model
    - we want to minimize ||T - T_hat||_{M^{-1}}^2 where M factors across modes
    
    the update for mode k is:
        A_k = (M_k^{-1/2} @ T_(k) @ (M_{-k}^{-1/2} @ A_{-k})^+)^T @ M_k^{-1/2}
    
    which reduces to solving weighted normal equations with the known metric.
    
    args:
        tenpy: tensor backend (numpy or ctf)
        T: input tensor to decompose
        A: initial factor matrices
        metric_factors: list of per-mode metric matrices M_k^{-1} from generator
        args: argument namespace with optimization parameters
    """
    
    def __init__(self, tenpy, T, A, metric_factors, args):
        self.tenpy = tenpy
        self.T = T
        self.A = A
        self.R = A[0].shape[1]
        self.order = len(A)
        self.sp = getattr(args, 'sp', 0)
        # metric_factors are precision matrices Σ_k^{-1} (inverse covariance)
        # This is verified by the generator which returns (U^+)^T @ U^+ ≈ (U^T U)^{-1}
        self.metric_factors = metric_factors  # list of Σ_k^{-1} matrices
        
        # precompute sqrt of metric factors for whitening
        # Σ_k^{-1} = (Σ_k^{-1/2})^T @ Σ_k^{-1/2}
        self.metric_sqrt = []
        for M_inv in metric_factors:
            # compute matrix square root via eigendecomposition
            eigvals, eigvecs = la.eigh(M_inv)
            # clip small/negative eigenvalues for numerical stability
            eigvals = np.maximum(eigvals, 1e-12)
            M_inv_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            self.metric_sqrt.append(M_inv_sqrt)
        
        # whiten the tensor once: T_white = T x_0 L_0 x_1 L_1 x ... x_{N-1} L_{N-1}
        # where L_k = metric_sqrt[k] = Σ_k^{-1/2}
        # This transforms the problem to min ||T_white - [[A_white]]||_F^2
        self.Tw = T.copy()
        for mode in range(self.order):
            # apply metric_sqrt[mode] along mode dimension
            # T_white = T_white x_mode L_mode
            L = self.metric_sqrt[mode]
            # Unfold tensor along mode: reshape to (mode_dim, other_dims_product)
            T_shape = list(self.Tw.shape)
            mode_dim = T_shape[mode]
            other_dims = [T_shape[i] for i in range(self.order) if i != mode]
            other_size = int(np.prod(other_dims))
            
            # Move mode to first position and reshape
            T_unfolded = np.moveaxis(self.Tw, mode, 0)
            T_unfolded = T_unfolded.reshape(mode_dim, other_size)
            
            # Apply whitening: L @ T_unfolded
            T_whitened = L @ T_unfolded
            
            # Reshape back and move mode back to original position
            T_whitened = T_whitened.reshape(mode_dim, *other_dims)
            self.Tw = np.moveaxis(T_whitened, 0, mode)
    
    def _build_mttkrp_einstr(self, mode):
        """build einsum string for weighted mttkrp at given mode."""
        # T indices: 'abc...'
        # output: mode_char + 'r'
        T_inds = ''.join([chr(ord('a') + i) for i in range(self.order)])
        out_inds = chr(ord('a') + mode) + 'r'
        
        # build contraction: metric_mode @ T contracted with weighted factors
        # for each mode != current: factor_mode @ metric_mode
        parts = []
        for i in range(self.order):
            if i != mode:
                parts.append(chr(ord('a') + i) + 'r')
        
        einstr = ','.join(parts) + ',' + T_inds + '->' + out_inds
        return einstr
    
    def step(self, Regu):
        """
        perform one sweep of weighted als updates using the known metric.
        
        for each mode k, solves the weighted normal equations:
            (sum_r weighted_khatri_rao) @ A_k^T = weighted_mttkrp
        
        args:
            Regu: regularization parameter
            
        returns:
            updated factor matrices
        """
        for mode in range(self.order):
            # compute whitened factors for all other modes
            # weighted_A[i] = Σ_i^{-1/2} @ A[i] (whitened factors)
            weighted_A = []
            for i in range(self.order):
                if i != mode:
                    weighted_A.append(self.metric_sqrt[i] @ self.A[i])
            
            # compute gramians of whitened factors
            gramians = []
            for wA in weighted_A:
                gramians.append(wA.T @ wA)
            
            # hadamard product of gramians
            S = gramians[0].copy()
            for g in gramians[1:]:
                S = S * g
            S = S + Regu * np.eye(self.R)
            
            # compute MTTKRP using whitened tensor and whitened factors
            # Since tensor is already whitened, we use weighted_A directly
            # and do NOT multiply rhs by metric_sqrt[mode] anymore
            mttkrp_factors = []
            for i in range(self.order):
                if i == mode:
                    mttkrp_factors.append(np.zeros((self.A[mode].shape[0], self.R)))
                else:
                    # find index in weighted_A
                    idx = sum(1 for j in range(i) if j != mode)
                    mttkrp_factors.append(weighted_A[idx])
            
            # compute MTTKRP on whitened tensor
            self.tenpy.MTTKRP(self.Tw, mttkrp_factors, mode)
            rhs = mttkrp_factors[mode]
            
            # solve the normal equations (no additional whitening needed)
            A_mode_white = solve_sys(self.tenpy, S, rhs)
            
            # unwhiten the result: A_mode = Σ_mode^{1/2} @ A_mode_white
            # where Σ_mode^{1/2} = (Σ_mode^{-1/2})^{-1}
            A_mode = la.solve(self.metric_sqrt[mode], A_mode_white)
            self.A[mode] = A_mode
        
        return self.A
    
    def compute_objective(self):
        """
        compute the mahalanobis objective ||T - T_hat||_{M^{-1}}^2.
        
        returns:
            scalar objective value
        """
        T_hat = cp_reconstruct(self.tenpy, self.A)
        diff = self.T - T_hat
        return mahalanobis_norm(self.tenpy, diff, self.metric_factors)

class CP_AMDM_Optimizer():
    """
    AMDM method for computing CP decomposition. The algorithm uses the general version of AMDM
    updates. The 
    
    Refer to the paper for details on how the optimization is carried out
    """
    def __init__(self,tenpy,T,A,args):
        self.tenpy = tenpy
        self.T = T
        self.A = A
        self.R = A[0].shape[1]
        self.sp = args.sp
        self.U = []
        self.sing = []
        self.VT = []
        self.thresh = args.thresh
        self.num_vals = args.num_vals
        self.nrm = None 
        self.update_svd()

    def reduce_vals(self):
        if self.num_vals > 0:
            self.num_vals -= 1

    def absorb_norm(self):
        self.A[0] = self.tenpy.einsum('r,ir->ir',self.nrm,self.A[0])

    def _einstr_builder(self, M, s, ii):
        ci = ""
        nd = M.ndim
        if len(s) != 1:
            ci = "R"
            nd = M.ndim - 1

        str1 = "".join([chr(ord('a') + j) for j in range(nd)]) + ci
        str2 = (chr(ord('a') + ii)) + "R"
        str3 = "".join([chr(ord('a') + j) for j in range(nd) if j != ii]) + "R"
        einstr = str1 + "," + str2 + "->" + str3
        return einstr

    def update_svd(self, i=None):
        if i is None:
            self.A,self.nrm = normalise(self.tenpy, self.A)
            for i in range(len(self.A)):
                U,s,VT = self.tenpy.svd(self.A[i])
                self.U.append(U)
                self.sing.append(s)
                self.VT.append(VT)
        else:
            self.A,self.nrm = normalise(self.tenpy, self.A, i)
            U,s,VT = self.tenpy.svd(self.A[i])
            self.U[i] = U
            self.sing[i] = s
            self.VT[i] = VT

    def compute_rhs(self,i):
        if self.thresh is not None:
            sing = np.where((self.sing[i][0] / self.sing[i]) < self.thresh, 
                1 / self.sing[i], self.sing[i])
        else:
            sing = np.concatenate((1 / self.sing[i][:self.num_vals], self.sing[i][self.num_vals:]))
        return self.tenpy.einsum('ir,r,rj->ij',self.U[i],sing,self.VT[i])

    def compute_rhs_lst(self,i):
        lst = []
        for j in range(len(self.A)):
            if j != i:
                if self.thresh is not None:
                    sing = np.where((self.sing[j][0] / self.sing[j]) < self.thresh, 
                        1 / self.sing[j], self.sing[j])
                else:
                    sing = np.concatenate((1 / self.sing[j][:self.num_vals],
                     self.sing[j][self.num_vals:]))
                lst.append(self.tenpy.einsum('ir,r,rj->ij',self.U[j],sing,self.VT[j]))
            else:
                lst.append(self.tenpy.zeros(self.A[i].shape))
        return lst

    def step(self,Regu):
        if not self.sp:
            q = queue.Queue()
        if self.sp:
            s = []
            for i in range(len(self.A)):
                lst = self.compute_rhs_lst(i)
                self.tenpy.MTTKRP(self.T,lst,i)
                if self.thresh is None and self.num_vals == self.A[i].shape[1]:
                    self.A[i] = lst[i]
                else:
                    self.A[i] = self._sp_solve(i,Regu,lst[i])
                self.update_svd(i)
        else:
            for i in range(len(self.A)):
                q.put(i)
            s = [(list(range(len(self.A))),self.T)]

            while not q.empty():
                i = q.get()
                while i not in s[-1][0]:
                    s.pop()
                    assert(len(s) >= 1)
                while len(s[-1][0]) != 1:
                    M = s[-1][1]
                    idx = s[-1][0].index(i)
                    ii = len(s[-1][0])-1
                    if idx == len(s[-1][0])-1:
                        ii = len(s[-1][0])-2
                    rh = self.compute_rhs(ii)
                    einstr = self._einstr_builder(M,s,ii)
                    N = self.tenpy.einsum(einstr,M,rh)

                    ss = s[-1][0][:]
                    ss.remove(ii)
                    s.append((ss,N))
                if self.thresh is None and self.num_vals == self.A[i].shape[1]:
                    self.A[i] = s[-1][1].copy()
                else:
                    self.A[i] = self._solve(i,Regu,s)
                self.update_svd(i)
        self.absorb_norm()
        return self.A

    def _sp_solve(self, i, Regu, g):
        lst = []
        for j in range(len(self.VT)):
            if self.thresh is not None:
                sing = np.where((self.sing[j][0] / self.sing[j]) < self.thresh, 
                1.0, self.sing[j])
            else:
                sing = np.concatenate((np.ones(self.num_vals), 
                    self.sing[j][self.num_vals:]))
            lst.append(self.tenpy.einsum('r,rj->rj',sing,self.VT[j]))
        return solve_sys(self.tenpy,
                        compute_lin_sysN(self.tenpy, lst, i, Regu),
                        g)

    def _solve(self, i, Regu, s):
        lst = []
        for j in range(len(self.VT)):
            if self.thresh is not None:
                sing = np.where((self.sing[j][0] / self.sing[j]) < self.thresh, 
                1.0, self.sing[j])
            else:
                sing = np.concatenate((np.ones(self.num_vals),
                 self.sing[j][self.num_vals:]))
            lst.append(self.tenpy.einsum('r,rj->rj',sing,self.VT[j]))
        return solve_sys(self.tenpy,
                        compute_lin_sysN(self.tenpy, lst, i, Regu),
                        s[-1][1])

