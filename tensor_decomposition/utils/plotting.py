"""
plotting utilities for visualizing tensor decomposition results.
generates comparison plots for als, amdm, and hybrid methods.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import colorsys


def lighten_color(color, amount=0.5):
    """
    lightens the given color by multiplying (1-luminosity) by the given amount.
    input can be matplotlib color string, hex string, or rgb tuple.
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def plot_convergence(
    iterations,
    values,
    labels=None,
    title='convergence plot',
    xlabel='iterations',
    ylabel='value',
    log_scale=True,
    save_path=None
):
    """
    plot convergence curves for one or more methods.
    
    args:
        iterations: list of iteration arrays (one per method)
        values: list of value arrays (one per method)
        labels: list of labels for legend
        title: plot title
        xlabel: x-axis label
        ylabel: y-axis label
        log_scale: whether to use log scale for y-axis
        save_path: if provided, save figure to this path
    """
    colors = ['orange', 'blue', 'red', 'green', 'purple']
    markers = ['.', '>', '<', 's', 'd']
    
    plt.figure(figsize=(10, 6))
    
    for i, (iters, vals) in enumerate(zip(iterations, values)):
        label = labels[i] if labels else f'method {i+1}'
        color = lighten_color(colors[i % len(colors)], 0.55)
        marker = markers[i % len(markers)]
        plt.plot(iters, vals, marker=marker, color=color, linewidth=2,
                 label=label, markersize=8)
    
    if log_scale:
        plt.yscale('log')
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend(prop={'size': 12})
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"[done] saved plot -> {save_path}")
    else:
        plt.show()


def plot_comparison_results(
    residuals,
    norm_mahalanobis,
    final_residuals,
    mean_residuals,
    std_residuals,
    size,
    rank,
    epsilon,
    alpha,
    save_path=None
):
    """
    generate a 2x2 subplot comparing different optimization methods.
    
    plots:
    1. iterations vs residual (log scale)
    2. iterations vs mahalanobis norm (log scale)
    3. scatter diagram of final residuals across runs
    4. error bar plot showing mean residuals with standard deviation
    """
    plt.figure(figsize=(14, 8))
    
    methods = ['ALS', 'AMDM', 'Hybrid']
    colors = ['orange', 'blue', 'red']

    # plot 1: iterations vs residual
    plt.subplot(2, 2, 1)
    for i, (res, method, color) in enumerate(zip(residuals, methods, colors)):
        iters = range(1, len(res) + 1)
        plt.plot(iters, res, marker='.', color=lighten_color(color, 0.55),
                 linewidth=2, label=f'rank={rank} {method}', markersize=10)
    
    plt.xlim(left=1, right=len(residuals[0]))
    plt.yscale('log')
    plt.grid(linestyle='--')
    plt.legend(prop={'size': 12})
    plt.title(f'noisy tensors: s={size}, ε={epsilon}, α={alpha}')
    plt.xlabel('iterations')
    plt.ylabel('absolute residual')

    # plot 2: iterations vs mahalanobis norm
    plt.subplot(2, 2, 2)
    for i, (nm, method, color) in enumerate(zip(norm_mahalanobis, methods, colors)):
        iters = range(1, len(nm) + 1)
        markers = ['.', '<', '<']
        plt.plot(iters, nm, marker=markers[i], color=lighten_color(color, 0.55),
                 linewidth=2, label=f'rank={rank} {method}', markersize=10)
    
    plt.xlim(left=1, right=len(norm_mahalanobis[0]))
    plt.yscale('log')
    plt.grid(linestyle='--')
    plt.legend(prop={'size': 12})
    plt.title(f'noisy tensors: s={size}, ε={epsilon}, α={alpha}')
    plt.xlabel('iterations')
    plt.ylabel(r"$||T-[A,B,C]||_{M^{-1}}$")

    # plot 3: scatter diagram of final residuals
    plt.subplot(2, 2, 3)
    num_runs = len(final_residuals[0])
    for i, (fr, method, color) in enumerate(zip(final_residuals, methods, colors)):
        plt.scatter(range(1, num_runs + 1), fr, color=color,
                    label=f'rank={rank} {method}', alpha=0.7)
    
    plt.xlim(left=1, right=num_runs)
    plt.yscale('log')
    plt.grid(linestyle='--')
    plt.legend(prop={'size': 12})
    plt.title('final residuals across runs')
    plt.xlabel('run index')
    plt.ylabel('final residual')

    # plot 4: error bar plot
    plt.subplot(2, 2, 4)
    fmts = ['-o', '-.', '-.']
    for i, (mean_res, std_res, method, color, fmt) in enumerate(
        zip(mean_residuals, std_residuals, methods, colors, fmts)
    ):
        iters = range(1, len(mean_res) + 1)
        plt.errorbar(iters, mean_res, yerr=std_res, fmt=fmt, capsize=5,
                     label=f"{method}: mean ± std", color=color)
    
    plt.xlim(left=1, right=len(mean_residuals[0]))
    plt.yscale('log')
    plt.grid(linestyle='--')
    plt.legend(prop={'size': 12})
    plt.title('residual with error bars')
    plt.xlabel('iterations')
    plt.ylabel('residual')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"[done] saved plot -> {save_path}")
    else:
        plt.show()

