"""
utility functions for generating tensors and initial factor matrices.
consolidates tensor generation and initialization utilities.
"""

import numpy as np


def generate_initial_guess(tenpy, tensor, args):
    """
    generate random initial factor matrices for tensor decomposition.
    
    creates random matrices of appropriate size based on the tensor
    dimensions and the specified approximation rank.
    
    args:
        tenpy: tensor backend (numpy or ctf)
        tensor: input tensor to decompose
        args: argument namespace with decomposition parameters
        
    returns:
        list of initial factor matrices
    """
    initial_factors = []
    
    tenpy.seed(args.seed)
    
    if args.decomposition == "CP":
        rank = getattr(args, 'R_app', args.R)
        for i in range(tensor.ndim):
            initial_factors.append(tenpy.random((tensor.shape[i], rank)))
    else:
        # tucker decomposition
        for i in range(tensor.ndim):
            initial_factors.append(tenpy.random((tensor.shape[i], args.hosvd_core_dim[i])))
    
    return initial_factors


def generate_tensor(tenpy, args):
    """
    generate or load an input tensor based on command line arguments.
    
    supports various tensor types including noisy tensors with different
    noise models, real-world tensors, and synthetic test tensors.
    
    args:
        tenpy: tensor backend (numpy or ctf)
        args: argument namespace with tensor parameters
        
    returns:
        dict with keys:
        - tensor_true: ground-truth clean tensor
        - tensor_noisy: tensor with noise (same as tensor_true if no noise model)
        - factors_true: list of ground-truth factor matrices U^(k) (None if not available)
        - cov_empirical: empirical covariance matrices (None if not available)
        - cov_pinv_empirical: pseudoinverse of empirical covariances (None if not available)  
        - m_empirical_pinv: metric factors for mahalanobis norm (None if not available)
        - sparsity_pattern: sparsity pattern if applicable (None otherwise)
    """
    # import here to avoid circular imports
    from tensor_decomposition.tensors import synthetic_tensors, real_tensors
    
    # initialize result dict with None defaults
    result = {
        'tensor_true': None,
        'tensor_noisy': None,
        'factors_true': None,
        'cov_empirical': None,
        'cov_pinv_empirical': None,
        'm_empirical_pinv': None,
        'sparsity_pattern': None,
    }
    
    if args.load_tensor != '':
        tensor = tenpy.load_tensor_from_file(args.load_tensor + 'tensor.npy')
        result['tensor_true'] = tensor
        result['tensor_noisy'] = tensor
        
    elif args.tensor == "random":
        tenpy.printf("[info] generating random tensor")
        [tensor, sparsity_pattern] = synthetic_tensors.rand(
            tenpy, args.order, args.s, args.R, args.sp_fraction, args.seed
        )
        result['tensor_true'] = tensor
        result['tensor_noisy'] = tensor
        result['sparsity_pattern'] = sparsity_pattern
        
    elif args.tensor == "noisy_tensor":
        dims = [args.s] * args.order
        if getattr(args, 'type_noisy_tensor', 'new_model') == "new_model":
            (tensor_true, tensor_noisy, factors_true, cov_empirical, 
             cov_pinv_empirical, m_empirical_pinv) = \
                synthetic_tensors.generate_tensor_with_noise_new_model(
                    tenpy, dims, args.R, args.k, args.epsilon, args.alpha, args.seed
                )
        else:
            (tensor_true, tensor_noisy, factors_true, cov_empirical, 
             cov_pinv_empirical, m_empirical_pinv) = \
                synthetic_tensors.generate_tensor_with_noise_old_model(
                    tenpy, dims, args.R, args.k, args.epsilon, args.alpha, args.seed
                )
        result['tensor_true'] = tensor_true
        result['tensor_noisy'] = tensor_noisy
        result['factors_true'] = factors_true
        result['cov_empirical'] = cov_empirical
        result['cov_pinv_empirical'] = cov_pinv_empirical
        result['m_empirical_pinv'] = m_empirical_pinv
            
    elif args.tensor == "MGH":
        tensor = tenpy.load_tensor_from_file("MGH-16.npy")
        tensor = tensor.reshape(
            tensor.shape[0] * tensor.shape[1],
            tensor.shape[2], tensor.shape[3], tensor.shape[4]
        )
        result['tensor_true'] = tensor
        result['tensor_noisy'] = tensor
        
    elif args.tensor == "SLEEP":
        tensor = tenpy.load_tensor_from_file("SLEEP-16.npy")
        tensor = tensor.reshape(
            tensor.shape[0] * tensor.shape[1],
            tensor.shape[2], tensor.shape[3], tensor.shape[4]
        )
        result['tensor_true'] = tensor
        result['tensor_noisy'] = tensor
        
    elif args.tensor == "random_col":
        (tensor_true, tensor_noisy, factors_true, cov_empirical,
         cov_pinv_empirical, m_empirical_pinv) = synthetic_tensors.collinearity_tensor(
            tenpy, args.s, args.order, args.R, args.k, args.epsilon, args.col, args.seed
        )
        result['tensor_true'] = tensor_true
        result['tensor_noisy'] = tensor_noisy
        result['factors_true'] = factors_true
        result['cov_empirical'] = cov_empirical
        result['cov_pinv_empirical'] = cov_pinv_empirical
        result['m_empirical_pinv'] = m_empirical_pinv
        
    elif args.tensor == "scf":
        tensor = real_tensors.get_scf_tensor(tenpy)
        result['tensor_true'] = tensor
        result['tensor_noisy'] = tensor
        
    elif args.tensor == "amino":
        tensor = real_tensors.amino_acids(tenpy)
        result['tensor_true'] = tensor
        result['tensor_noisy'] = tensor
        
    elif args.tensor == "negrandom":
        tenpy.printf("[info] generating random tensor with negative entries")
        [tensor, sparsity_pattern] = synthetic_tensors.neg_rand(
            tenpy, args.order, args.s, args.R, args.sp_fraction, args.seed
        )
        result['tensor_true'] = tensor
        result['tensor_noisy'] = tensor
        result['sparsity_pattern'] = sparsity_pattern
        
    elif args.tensor == "randn":
        tenpy.printf("[info] generating random tensor with normal entries")
        [tensor, sparsity_pattern] = synthetic_tensors.randn(
            tenpy, args.order, args.s, args.R, args.sp_fraction, args.seed
        )
        result['tensor_true'] = tensor
        result['tensor_noisy'] = tensor
        result['sparsity_pattern'] = sparsity_pattern

    if result['tensor_noisy'] is not None:
        tenpy.printf(f"[info] input tensor shape: {result['tensor_noisy'].shape}")
    
    return result

