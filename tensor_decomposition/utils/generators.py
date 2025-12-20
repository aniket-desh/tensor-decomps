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
        tuple of (tensor_true, tensor, sparsity_pattern, cov_empirical,
                  cov_pinv_empirical, m_empirical_pinv)
    """
    # import here to avoid circular imports
    from tensor_decomposition.tensors import synthetic_tensors, real_tensors
    
    tensor_true = None
    tensor = None
    sparsity_pattern = None
    cov_empirical = None
    cov_pinv_empirical = None
    m_empirical_pinv = None
    
    if args.load_tensor != '':
        tensor = tenpy.load_tensor_from_file(args.load_tensor + 'tensor.npy')
        tensor_true = tensor
        
    elif args.tensor == "random":
        tenpy.printf("[info] generating random tensor")
        [tensor, sparsity_pattern] = synthetic_tensors.rand(
            tenpy, args.order, args.s, args.R, args.sp_fraction, args.seed
        )
        tensor_true = tensor
        
    elif args.tensor == "noisy_tensor":
        dims = [args.s] * args.order
        if getattr(args, 'type_noisy_tensor', 'new_model') == "new_model":
            (tensor_true, tensor, cov_empirical, cov_pinv_empirical,
             m_empirical_pinv) = synthetic_tensors.generate_tensor_with_noise_new_model(
                tenpy, dims, args.R, args.k, args.epsilon, args.alpha, args.seed
            )
        else:
            (tensor_true, tensor, cov_empirical, cov_pinv_empirical, _, _,
             m_empirical_pinv, _) = synthetic_tensors.generate_tensor_with_noise_old_model(
                tenpy, dims, args.R, args.k, args.epsilon, args.alpha, args.seed
            )
            
    elif args.tensor == "MGH":
        tensor = tenpy.load_tensor_from_file("MGH-16.npy")
        tensor = tensor.reshape(
            tensor.shape[0] * tensor.shape[1],
            tensor.shape[2], tensor.shape[3], tensor.shape[4]
        )
        tensor_true = tensor
        
    elif args.tensor == "SLEEP":
        tensor = tenpy.load_tensor_from_file("SLEEP-16.npy")
        tensor = tensor.reshape(
            tensor.shape[0] * tensor.shape[1],
            tensor.shape[2], tensor.shape[3], tensor.shape[4]
        )
        tensor_true = tensor
        
    elif args.tensor == "random_col":
        dims = [args.s] * args.order
        result = synthetic_tensors.collinearity_tensor(
            tenpy, args.s, args.order, args.R, args.k, args.epsilon, args.col, args.seed
        )
        if len(result) == 6:
            (tensor_true, sparsity_pattern, tensor, cov_empirical,
             cov_pinv_empirical, m_empirical_pinv) = result
        else:
            # fallback for different signature
            tensor, sparsity_pattern = result[0], result[1]
            tensor_true = tensor
        
    elif args.tensor == "scf":
        tensor = real_tensors.get_scf_tensor(tenpy)
        tensor_true = tensor
        
    elif args.tensor == "amino":
        tensor = real_tensors.amino_acids(tenpy)
        tensor_true = tensor
        
    elif args.tensor == "negrandom":
        tenpy.printf("[info] generating random tensor with negative entries")
        [tensor, sparsity_pattern] = synthetic_tensors.neg_rand(
            tenpy, args.order, args.s, args.R, args.sp_fraction, args.seed
        )
        tensor_true = tensor
        
    elif args.tensor == "randn":
        tenpy.printf("[info] generating random tensor with normal entries")
        [tensor, sparsity_pattern] = synthetic_tensors.randn(
            tenpy, args.order, args.s, args.R, args.sp_fraction, args.seed
        )
        tensor_true = tensor

    if tensor is not None:
        tenpy.printf(f"[info] input tensor shape: {tensor.shape}")
    
    return tensor_true, tensor, sparsity_pattern, cov_empirical, cov_pinv_empirical, m_empirical_pinv

