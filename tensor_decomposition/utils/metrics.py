"""
evaluation metrics for tensor decomposition factor recovery.

provides metrics for comparing estimated factor matrices against ground-truth,
handling the inherent permutation and scaling ambiguities in CP decomposition.
"""

import numpy as np
import numpy.linalg as la
from scipy.optimize import linear_sum_assignment


def cosine_similarity(a, b):
    """
    compute cosine similarity between two vectors.
    
    args:
        a: first vector
        b: second vector
        
    returns:
        cosine similarity in [-1, 1]
    """
    norm_a = la.norm(a)
    norm_b = la.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)


def factor_match_score(factors_true, factors_est, return_permutation=False):
    """
    compute factor match score (fms) between true and estimated factors.
    
    the fms handles permutation ambiguity by finding the optimal column
    matching using the hungarian algorithm, and handles scaling ambiguity
    by using cosine similarity.
    
    fms = (1/R) * sum_r |cos(u_true^r, u_est^{pi(r)})|
    
    where pi is the optimal permutation that maximizes the total score.
    the absolute value handles sign ambiguity.
    
    args:
        factors_true: list of N ground-truth factor matrices [U_1, ..., U_N]
        factors_est: list of N estimated factor matrices [U_1_est, ..., U_N_est]
        return_permutation: if True, also return the optimal permutation
        
    returns:
        fms: scalar in [0, 1], where 1 means perfect recovery
        permutation: (optional) the optimal column permutation
    """
    N = len(factors_true)
    R = factors_true[0].shape[1]
    R_est = factors_est[0].shape[1]
    
    if R != R_est:
        raise ValueError(f"rank mismatch: true has {R} columns, estimated has {R_est}")
    
    # compute cost matrix based on product of cosine similarities across modes
    # cost[i, j] = product over modes of |cos(u_true_mode[:, i], u_est_mode[:, j])|
    cost_matrix = np.ones((R, R))
    for mode in range(N):
        for i in range(R):
            for j in range(R):
                cos_sim = cosine_similarity(factors_true[mode][:, i], factors_est[mode][:, j])
                cost_matrix[i, j] *= abs(cos_sim)
    
    # use hungarian algorithm to find optimal permutation (maximize similarity)
    # linear_sum_assignment minimizes, so we negate
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    
    # compute fms as average of matched similarities
    fms = np.mean(cost_matrix[row_ind, col_ind])
    
    if return_permutation:
        return fms, col_ind
    return fms


def aligned_factor_error(factors_true, factors_est, normalize=True):
    """
    compute aligned frobenius error between true and estimated factors.
    
    first finds the optimal permutation and scaling, then computes the
    frobenius norm of the difference per mode.
    
    args:
        factors_true: list of N ground-truth factor matrices
        factors_est: list of N estimated factor matrices  
        normalize: if True, normalize columns before comparison
        
    returns:
        errors: list of N per-mode frobenius errors
        total_error: sum of per-mode errors
        permutation: the optimal column permutation used
    """
    N = len(factors_true)
    R = factors_true[0].shape[1]
    
    # find optimal permutation using fms
    _, permutation = factor_match_score(factors_true, factors_est, return_permutation=True)
    
    # permute estimated factors
    factors_est_perm = [F[:, permutation] for F in factors_est]
    
    errors = []
    for mode in range(N):
        U_true = factors_true[mode].copy()
        U_est = factors_est_perm[mode].copy()
        
        if normalize:
            # normalize columns
            for r in range(R):
                norm_true = la.norm(U_true[:, r])
                norm_est = la.norm(U_est[:, r])
                if norm_true > 1e-12:
                    U_true[:, r] /= norm_true
                if norm_est > 1e-12:
                    U_est[:, r] /= norm_est
        
        # handle sign ambiguity: flip sign if it reduces error
        for r in range(R):
            if la.norm(U_true[:, r] + U_est[:, r]) < la.norm(U_true[:, r] - U_est[:, r]):
                U_est[:, r] = -U_est[:, r]
        
        error = la.norm(U_true - U_est, 'fro')
        errors.append(error)
    
    return errors, sum(errors), permutation


def congruence_coefficient(factors_true, factors_est):
    """
    compute tucker's congruence coefficient for factor recovery.
    
    similar to fms but uses the traditional congruence coefficient formula.
    handles permutation via hungarian algorithm.
    
    args:
        factors_true: list of N ground-truth factor matrices
        factors_est: list of N estimated factor matrices
        
    returns:
        cc: congruence coefficient in [0, 1]
    """
    return factor_match_score(factors_true, factors_est)


def relative_factor_error(factors_true, factors_est):
    """
    compute relative error in factor recovery.
    
    computes ||U_true - P @ U_est @ D||_F / ||U_true||_F for each mode,
    where P is a permutation matrix and D is a diagonal scaling matrix
    chosen to minimize the error.
    
    args:
        factors_true: list of N ground-truth factor matrices
        factors_est: list of N estimated factor matrices
        
    returns:
        rel_errors: list of N per-mode relative errors
        mean_rel_error: average relative error across modes
    """
    N = len(factors_true)
    R = factors_true[0].shape[1]
    
    # find optimal permutation
    _, permutation = factor_match_score(factors_true, factors_est, return_permutation=True)
    
    rel_errors = []
    for mode in range(N):
        U_true = factors_true[mode]
        U_est = factors_est[mode][:, permutation]
        
        # find optimal scaling per column
        U_est_scaled = U_est.copy()
        for r in range(R):
            u_t = U_true[:, r]
            u_e = U_est[:, r]
            
            # optimal scale: argmin_s ||u_t - s*u_e||^2 = (u_t . u_e) / ||u_e||^2
            denom = np.dot(u_e, u_e)
            if denom > 1e-12:
                scale = np.dot(u_t, u_e) / denom
                U_est_scaled[:, r] = scale * u_e
        
        error = la.norm(U_true - U_est_scaled, 'fro')
        norm_true = la.norm(U_true, 'fro')
        
        rel_error = error / norm_true if norm_true > 1e-12 else error
        rel_errors.append(rel_error)
    
    return rel_errors, np.mean(rel_errors)

