# tensor decomposition library (wip)

a python library for efficient tensor decomposition, featuring multiple optimization methods for cp (canonical polyadic) and tucker decompositions.

## overview

this repository implements:

- **alternating least squares (als)**: standard iterative method for cp and tucker decompositions
- **pairwise perturbation als (pp-als)**: accelerated als variant using low-rank updates  
- **fast nonlinear least squares (nls)**: gauss-newton method with preconditioned conjugate gradient
- **alternating mahalanobis distance minimization (amdm)**: optimization using mahalanobis norm for noisy tensors
- **hybrid methods**: combinations of als and amdm for improved convergence

supports both numpy (sequential) and [cyclops tensor framework (ctf)](https://github.com/cyclops-community/ctf) (distributed parallel) backends.

## installation

```bash
pip install .
```

for ctf backend support, install ctf separately following their documentation.

## quick start

```python
import tensor_decomposition as td

# get a backend (numpy or ctf)
tenpy = td.get_backend('numpy')

# create a random tensor
tensor = tenpy.random((10, 10, 10))

# initialize random factor matrices
rank = 5
factors = [tenpy.random((10, rank)) for _ in range(3)]

# create an als optimizer
class Args:
    sp = 0
    fast_residual = 0
    
opt = td.CP_DTALS_Optimizer(tenpy, tensor, factors, Args())

# run optimization
for i in range(100):
    factors = opt.step(regularization=1e-6)
    res = td.get_residual(tenpy, tensor, factors)
    print(f"[info] iter={i} | residual={res:.2e}")
```

## project structure

```
tensor_decomposition/
├── __init__.py            # main api - import tensor_decomposition as td
├── als/                   # base als optimizer classes
│   └── ALS_optimizer.py   # DTALS_base, PPALS_base
├── backend/               # numpy and ctf backends
│   ├── __init__.py        # get_backend() function
│   ├── numpy_ext.py
│   └── ctf_ext.py
├── CPD/                   # cp decomposition algorithms
│   ├── __init__.py
│   ├── standard_ALS.py    # CP_DTALS_Optimizer, CP_PPALS_Optimizer
│   ├── NLS.py             # CP_fastNLS_Optimizer
│   ├── mahalanobis.py     # CP_AMDM_Optimizer
│   └── common_kernels.py  # get_residual, compute_condition_number, etc.
├── Tucker/                # tucker decomposition algorithms
├── tensors/               # tensor generation utilities
│   ├── synthetic_tensors.py
│   └── real_tensors.py
└── utils/                 # utilities
    ├── __init__.py        # exports all utilities
    ├── arg_defs.py        # argument parsing helpers
    ├── generators.py      # generate_tensor, generate_initial_guess
    ├── plotting.py        # plot_convergence, plot_comparison_results
    └── utils.py           # save_decomposition_results
```

## centralized imports

the library provides a clean api through the main module:

```python
import tensor_decomposition as td

# backend
tenpy = td.get_backend('numpy')  # or 'ctf'

# optimizers
td.CP_DTALS_Optimizer    # standard als
td.CP_PPALS_Optimizer    # pairwise perturbation als
td.CP_fastNLS_Optimizer  # gauss-newton nls
td.CP_AMDM_Optimizer     # mahalanobis distance minimization

# kernels
td.get_residual          # compute residual norm
td.get_residual3         # residual for order-3 tensors
td.compute_condition_number

# utilities
td.generate_initial_guess
td.generate_tensor
td.save_decomposition_results
td.plot_convergence
td.plot_comparison_results
```

## experiments

all experiments are in the `experiments/` directory:

```bash
cd experiments
pip install -r requirements.txt
```

### running cp decomposition with als

```bash
python run_als.py --help
```

### running cp decomposition with nls

```bash
python run_nls.py --s 32 --R 5 --num-iter 50 --tlib numpy
```

### matrix multiplication tensor experiments

```bash
python matmul.py --m1 3 --m2 3 --m3 3 --R 23 --method HYB --tries 10
```

## logging format

all experiments use a consistent logging format:

| tag | purpose | example |
|-----|---------|---------|
| `[info]` | status updates | `[info] iter=10 \| residual=1.23e-04` |
| `[trial XX/YY]` | per-trial progress | `[trial 03/15] starting optimization` |
| `[summary]` | aggregate results | `[summary] als method took 2.45s overall` |
| `[done]` | completion | `[done] results saved -> ./results/output.csv` |
| `[error]` | errors | `[error] tucker not supported via gauss-newton` |

## backends

### numpy backend
fast sequential execution, recommended for development and smaller tensors.

### ctf backend  
distributed parallel execution using mpi. required for large-scale experiments on clusters.

```bash
mpirun -np 4 python run_nls.py --tlib ctf --s 128 --R 20
```

## acknowledgements

this code was extended by [Aniket Deshpande](https://aniketdeshpande.com). the original codebase was written by [Maryam Dehghan](https://scholar.google.com/citations?hl=en&user=vg8czIcAAAAJ&view_op=list_works&sortby=pubdate)
