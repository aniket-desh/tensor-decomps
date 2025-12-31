import argparse, itertools, time, os
from dataclasses import dataclass
from os.path import dirname, join
import numpy as np
import pandas as pd
import tensor_decomposition as td

# generator and init
from tensor_decomposition.utils.generators import generate_tensor, generate_initial_guess

# optimizers
from tensor_decomposition.cpd.standard_ALS import CP_DTALS_Optimizer
from tensor_decomposition.cpd.mahalanobis import CP_AMDM_Optimizer, CP_AMDM_MLE_Optimizer

# kernels and metrics
from tensor_decomposition.cpd.common_kernels import cp_reconstruct, get_residual, mahalanobis_norm
from tensor_decomposition.utils.metrics import factor_match_score

from typing import Type

# set up paths relative to script location
PARENT_DIR = dirname(dirname(__file__))  # go up from experiments/mle/ to experiments/
RESULTS_DIR = join(PARENT_DIR, 'results')


def _parse_list(x, cast=float):
    # accept '0.1,0.2,1.0' or '0.1 0.2 1.0'
    if x is None:
        return []
    parts = [p.strip() for p in x.replace(',', ' ').split()]
    return [cast(p) for p in parts if p]

def _copy_factors(factors):
    # factors is a list of matrices
    return [A.copy() for A in factors]

def _run_optimizer(opt, n_iter, regularization=1e-6):
    # optimizer.step(Regu) takes regularization as positional argument
    for _ in range(n_iter):
        result = opt.step(regularization)
        # some optimizers return (factors, restart_flag), others just return factors
        if isinstance(result, tuple):
            factors = result[0]
        else:
            factors = result
    return factors

def _eval_all(tenpy, tensor_true, tensor_noisy, factors_true, factors_hat, metric_factors_oracle):
    # returns a dict of eval metrics
    # we compute both:
    # - fit to noisy tensor (what opt minimizes)
    # - error to true tensor (what we care abt in synthetic recovery)
    # - factor match score (permutation and scale invariant)
    # - mahalanobis norms using oracle metric
    T_hat = cp_reconstruct(tenpy, factors_hat)

    # fit to noisy observation
    resid_noisy = tensor_noisy - T_hat
    frob_resid_noisy = tenpy.vecnorm(resid_noisy)
    mahal_resid_noisy = mahalanobis_norm(tenpy, resid_noisy, metric_factors_oracle)

    # error to ground truth signal
    err_true = tensor_true - T_hat
    frob_err_true = tenpy.vecnorm(err_true)
    mahal_err_true = mahalanobis_norm(tenpy, err_true, metric_factors_oracle)

    # factor recovery
    fms = factor_match_score(factors_true, factors_hat)

    return dict(
        frob_resid_noisy=float(frob_resid_noisy),
        mahal_resid_noisy=float(mahal_resid_noisy),
        frob_err_true=float(frob_err_true),
        mahal_err_true=float(mahal_err_true),
        fms=float(fms),
    )

def plot_from_csv(csv_path: str,
                  out_dir: str = None,
                  facet_row: str = "alpha",
                  facet_col: str = "k",
                  metrics=("fms", "mahal_err_true"),
                  logx: bool = True,
                  logy: bool = False):
    """
    read the CSV produced by Experiment A and save seaborn plots.

    default:
      - fms vs eps (primary "recover U" metric)
      - mahal_err_true vs eps (model-matched error diagnostic)

    faceting:
      - row = alpha (anisotropy / conditioning control)
      - col = k (CLT-ness / number of perturbation terms)
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    if out_dir is None:
        out_dir = join(RESULTS_DIR, 'plots')
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    # make sure key columns are numeric
    for c in ["eps", "alpha", "k"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    sns.set_theme(style="whitegrid")

    # seaborn API changed, newer versions use errorbar=..., older use ci=...
    def _lineplot(**kwargs):
        ax = kwargs.pop("ax", None)
        try:
            # seaborn >= 0.12-ish
            return sns.lineplot(**kwargs, estimator="mean", errorbar="sd", ax=ax)
        except TypeError:
            # seaborn <= 0.11-ish
            return sns.lineplot(**kwargs, estimator="mean", ci="sd", ax=ax)

    for metric in metrics:
        if metric not in df.columns:
            print(f"[plot] skipping '{metric}' (not in CSV columns)")
            continue

        # relplot makes faceting dead simple for (row, col) grids
        # kind="line" uses lineplot under the hood.
        try:
            g = sns.relplot(
                data=df,
                x="eps",
                y=metric,
                hue="method",
                style="method",
                row=facet_row if facet_row in df.columns else None,
                col=facet_col if facet_col in df.columns else None,
                kind="line",
                estimator="mean",
                errorbar="sd",   # seaborn newer
                markers=True,
                dashes=False,
                facet_kws=dict(sharex=True, sharey=True),
            )
        except TypeError:
            # fallback for older seaborn
            g = sns.relplot(
                data=df,
                x="eps",
                y=metric,
                hue="method",
                style="method",
                row=facet_row if facet_row in df.columns else None,
                col=facet_col if facet_col in df.columns else None,
                kind="line",
                estimator="mean",
                ci="sd",         # seaborn older
                markers=True,
                dashes=False,
                facet_kws=dict(sharex=True, sharey=True),
            )

        if logx:
            for ax in g.axes.flat:
                ax.set_xscale("log")
        if logy:
            for ax in g.axes.flat:
                ax.set_yscale("log")

        g.set_axis_labels("epsilon (noise scale)", metric)
        g.fig.suptitle(f"Experiment A: {metric} vs epsilon", y=1.02)

        out_path = os.path.join(out_dir, f"mle_comparison_{metric}.png")
        plt.savefig(out_path, bbox_inches="tight", dpi=200)
        plt.close(g.fig)
        print(f"[plot] wrote {out_path}")

    # optional: quick "collapsed" view with alpha/k fixed to their most common values
    if facet_row in df.columns and facet_col in df.columns:
        alpha_mode = df[facet_row].mode().iloc[0]
        k_mode = df[facet_col].mode().iloc[0]
        d2 = df[(df[facet_row] == alpha_mode) & (df[facet_col] == k_mode)]

        if len(d2) > 0 and "fms" in d2.columns:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(7, 4))
            _lineplot(data=d2, x="eps", y="fms", hue="method", style="method", markers=True, dashes=False)
            if logx:
                plt.xscale("log")
            plt.title(f"Factor Match Score vs eps (alpha={alpha_mode}, k={k_mode})")
            plt.xlabel("Epsilon (noise scale)")
            plt.ylabel("Factor Match Score")
            out_path = os.path.join(out_dir, f"mle_comparison_fms_collapsed.png")
            plt.savefig(out_path, bbox_inches="tight", dpi=200)
            plt.close()
            print(f"[plot] wrote {out_path}")

def main():
    p = argparse.ArgumentParser()

    # tensor and model parameters
    p.add_argument("--tlib", type=str, default="numpy", choices=["numpy", "ctf"])
    p.add_argument("--order", type=int, default=3)
    p.add_argument("--s", type=int, default=30, help="mode size (uses same size for each mode)")
    p.add_argument("--R", type=int, default=5, help="CP rank")
    p.add_argument("--k", type=int, default=20, help="noise averaging parameter in new model")
    p.add_argument("--alpha", type=float, default=1.2, help="controls singular value decay / anisotropy")
    p.add_argument("--epsilon", type=float, default=0.1, help="noise scale")
    
    # arguments needed by generate_tensor and generate_initial_guess
    p.add_argument("--load_tensor", type=str, default="", help="Path to load tensor from file (empty string = don't load)")
    p.add_argument("--decomposition", type=str, default="CP", choices=["CP", "Tucker"], help="Decomposition type")
    p.add_argument("--sp_fraction", type=float, default=0.0, help="Sparsity fraction for random tensors")
    
    # arguments needed by optimizers
    p.add_argument("--sp", type=int, default=0, help="Sparse decomposition flag (default: 0)")
    p.add_argument("--fast_residual", type=int, default=0, help="Enable fast residual calculation (default: 0)")
    p.add_argument("--tol_restart_dt", type=float, default=0.01, help="Tolerance for dimension tree restart in PP optimizer")
    p.add_argument("--thresh", type=float, default=None, help="Threshold for ratio of inverting singular values (AMDM)")
    p.add_argument("--num_vals", type=int, default=None, help="Number of singular values inverted (AMDM, default: None)")

    # sweeps
    p.add_argument("--eps_grid", type=str, default="0.01,0.03,0.1,0.3,1.0")
    p.add_argument("--alpha_grid", type=str, default="1.01,1.05,1.1,1.2,1.5")
    p.add_argument("--k_grid", type=str, default="5,10,20,50")

    # optimization
    p.add_argument("--n_iter", type=int, default=50)
    p.add_argument("--regularization", type=float, default=1e-6)
    p.add_argument("--n_trials", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)

    # output
    p.add_argument("--out_csv", type=str, default=join(RESULTS_DIR, "mle_comparison.csv"))

    # plotting
    p.add_argument("--plot", action="store_true", help="If set, make seaborn plots from the CSV after running.")
    p.add_argument("--plot_only", action="store_true", help="If set, skip running; only plot from --out_csv.")
    p.add_argument("--plot_dir", type=str, default=join(RESULTS_DIR, "plots"))
    p.add_argument("--plot_metrics", type=str, default="fms,mahal_err_true",
                  help="Comma-separated list of metrics to plot. e.g. fms,mahal_err_true,frob_err_true")
    p.add_argument("--facet_row", type=str, default="alpha")
    p.add_argument("--facet_col", type=str, default="k")
    p.add_argument("--logx", action="store_true", help="Log-scale x-axis (epsilon).")
    p.add_argument("--logy", action="store_true", help="Log-scale y-axis (metric).")

    args = p.parse_args()

    if args.plot_only:
        metrics = [m.strip() for m in args.plot_metrics.split(",") if m.strip()]
        plot_from_csv(
            csv_path=args.out_csv,
            out_dir=args.plot_dir,
            facet_row=args.facet_row,
            facet_col=args.facet_col,
            metrics=metrics,
            logx=args.logx,
            logy=args.logy,
        )
        return

    tenpy = td.get_backend(args.tlib)

    eps_grid = _parse_list(args.eps_grid, float)
    alpha_grid = _parse_list(args.alpha_grid, float)
    k_grid = _parse_list(args.k_grid, int)

    # set num_vals to rank if not specified (required by CP_AMDM_Optimizer)
    if args.num_vals is None:
        args.num_vals = args.R

    rows = []

    # force new noise model
    args.tensor = "noisy_tensor"
    args.type_noisy_tensor = "new_model"

    for (eps, alpha, k) in itertools.product(eps_grid, alpha_grid, k_grid):
        for t in range(args.n_trials):
            trial_seed = args.seed + 100000 * t + 1000 * int(100 * eps) + 10 * int(100 * (alpha - 1)) + k
            np.random.seed(trial_seed)

            # sample from probabilistic model
            args.epsilon = float(eps)
            args.alpha = float(alpha)
            args.k = int(k)

            d = generate_tensor(tenpy, args)
            tensor_true = d["tensor_true"]
            tensor_noisy = d["tensor_noisy"]
            factors_true = d["factors_true"]

            # oracle per-mode precision factors (Kronecker Σ0^{-1})
            metric_factors_oracle = d["m_empirical_pinv"]

            # init
            factors0 = generate_initial_guess(tenpy, tensor_noisy, args)

            # (1) ALS
            f0 = _copy_factors(factors0)
            opt_als = CP_DTALS_Optimizer(tenpy, tensor_noisy, f0, args)
            t0 = time.time()
            factors_als = _run_optimizer(opt_als, args.n_iter, regularization=args.regularization)
            dt = time.time() - t0
            m_als = _eval_all(tenpy, tensor_true, tensor_noisy, factors_true, factors_als, metric_factors_oracle)
            rows.append(dict(method="ALS", eps=eps, alpha=alpha, k=k, trial=t, seed=trial_seed, wall_sec=dt, **m_als))

            # (2) Oracle-AMDM: uses known oracle metric factors
            # CP_AMDM_MLE_Optimizer(tenpy, T, A, metric_factors, args)
            f0 = _copy_factors(factors0)
            opt_oracle = CP_AMDM_MLE_Optimizer(tenpy, tensor_noisy, f0, metric_factors_oracle, args)
            t0 = time.time()
            factors_oracle = _run_optimizer(opt_oracle, args.n_iter, regularization=args.regularization)
            dt = time.time() - t0
            m_oracle = _eval_all(tenpy, tensor_true, tensor_noisy, factors_true, factors_oracle, metric_factors_oracle)
            rows.append(dict(method="Oracle-AMDM", eps=eps, alpha=alpha, k=k, trial=t, seed=trial_seed, wall_sec=dt, **m_oracle))

            # (3) Standard AMDM: estimates metric from current factors
            # CP_AMDM_Optimizer(tenpy, T, A, args)
            f0 = _copy_factors(factors0)
            opt_amdm = CP_AMDM_Optimizer(tenpy, tensor_noisy, f0, args)
            t0 = time.time()
            factors_amdm = _run_optimizer(opt_amdm, args.n_iter, regularization=args.regularization)
            dt = time.time() - t0
            m_amdm = _eval_all(tenpy, tensor_true, tensor_noisy, factors_true, factors_amdm, metric_factors_oracle)
            rows.append(dict(method="AMDM", eps=eps, alpha=alpha, k=k, trial=t, seed=trial_seed, wall_sec=dt, **m_amdm))

            print(f"[done] eps={eps} alpha={alpha} k={k} trial={t}")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"[done] wrote {args.out_csv}")

    if args.plot:
        metrics = [m.strip() for m in args.plot_metrics.split(",") if m.strip()]
        plot_from_csv(
            csv_path=args.out_csv,
            out_dir=args.plot_dir,
            facet_row=args.facet_row,
            facet_col=args.facet_col,
            metrics=metrics,
            logx=args.logx,
            logy=args.logy,
        )


if __name__ == "__main__":
    main()
