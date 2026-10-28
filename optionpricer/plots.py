"""
Plots for the streamlit app and the README.

Uses Figure directly instead of pyplot so it works in streamlit.
"""
import numpy as np
from matplotlib.figure import Figure

from .binomial import binomial_price
from .black_scholes import bs_greeks, bs_price
from .monte_carlo import mc_price

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREY = "#6b6a66"
GRID = "#e4e3df"

def _style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, loc="left", fontsize=12, pad=10)
    ax.set_xlabel(xlabel, color=GREY)
    ax.set_ylabel(ylabel, color=GREY)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=GREY)

def binomial_convergence(S, K, T, r, sigma, option_type="call", q=0.0, max_steps=300):
    steps = np.arange(5, max_steps + 1)
    tree = [binomial_price(S, K, T, r, sigma, option_type, q, steps=int(n)) for n in steps]
    exact = bs_price(S, K, T, r, sigma, option_type, q)
    
    fig = Figure(figsize=(7, 3.8), layout="tight")
    ax = fig.subplots()
    ax.plot(steps, tree, color=BLUE, linewidth=1.5, label="CRR tree")
    ax.axhline(exact, color=ORANGE, linewidth=2, label=f"Black-Scholes = {exact:.4f}")
    _style(ax, "Binomial tree vs Black-Scholes (note the odd/even zig-zag)",
           "Number of steps", "Price")
    ax.legend(frameon=False)
    return fig

def mc_convergence(S, K, T, r, sigma, option_type="call", q=0.0, seed=0,
                   antithetic=True, control_variate=False):
    n_paths = np.unique(np.logspace(2, 6, 25).astype(int))
    runs = [mc_price(S, K, T, r, sigma, option_type, q, n_path=int(n), seed=seed,
                     antithetic=antithetic, control_variate=control_variate)
            for n in n_paths]
    prices = np.array([run.price for run in runs])
    half_width = 1.96 * np.array([run.std_error for run in runs])
    exact = bs_price(S, K, T, r, sigma, option_type, q)

    fig = Figure(figsize=(7, 3.8), layout="tight")
    ax = fig.subplots()
    ax.fill_between(n_paths, prices - half_width, prices + half_width, color=BLUE,
                    alpha=0.18, linewidth=0, label="95% CI")
    ax.plot(n_paths, prices, color=BLUE, linewidth=2, marker="o", markersize=4,
            label="MC estimate")
    ax.axhline(exact, color=ORANGE, linewidth=2, label=f"Black-Scholes = {exact:.4f}")
    ax.set_xscale("log")
    _style(ax, "Monte Carlo convergence", "Number of paths (log scale)", "Price")
    ax.legend(frameon=False)
    return fig

def mc_variance_reduction(S, K, T, r, sigma, option_type="call", q=0.0, seed=0):
    n_paths = np.unique(np.logspace(2, 6, 25).astype(int))
    exact = bs_price(S, K, T, r, sigma, option_type, q)
    settings = [
        ("Plain MC", ORANGE, dict(antithetic=False, control_variate=False)),
        ("Antithetic + control variate", BLUE, dict(antithetic=True, control_variate=True)),
    ]

    fig = Figure(figsize=(7, 3.8), layout="tight")
    ax = fig.subplots()
    for label, color, kwargs in settings:
        runs = [mc_price(S, K, T, r, sigma, option_type, q, n_path=int(n), seed=seed,
                         **kwargs) for n in n_paths]
        err = np.array([run.price for run in runs]) - exact
        half_width = 1.96 * np.array([run.std_error for run in runs])
        ax.fill_between(n_paths, err - half_width, err + half_width, color=color,
                        alpha=0.16, linewidth=0)
        ax.plot(n_paths, err, color=color, linewidth=2, marker="o", markersize=4,
                label=f"{label} (95% CI)")
    ax.axhline(0, color=GREY, linewidth=1)
    ax.set_xscale("log")
    _style(ax, "Variance reduction, same number of paths",
           "Number of paths (log scale)", "Error vs Black-Scholes")
    ax.legend(frameon=False)
    return fig


def price_vs_spot(S, K, T, r, sigma, option_type="call", q=0.0):
    spots = np.linspace(0.5 * K, 1.5 * K, 121)
    prices = [bs_price(s, K, T, r, sigma, option_type, q) for s in spots]
    if option_type == "call":
        intrinsic = np.maximum(spots - K, 0)
    else:
        intrinsic = np.maximum(K - spots, 0)

    fig = Figure(figsize=(7, 3.8), layout="tight")
    ax = fig.subplots()
    ax.plot(spots, intrinsic, color=GREY, linewidth=1.5, linestyle="--", label="Payoff at expiry")
    ax.plot(spots, prices, color=BLUE, linewidth=2, label=f"Value today (T = {T:g}y)")
    ax.axvline(S, color=ORANGE, linewidth=1.5, label=f"Spot = {S:g}")
    _style(ax, f"{option_type.title()} value vs spot (gap = time value)", "Spot", "Value")
    ax.legend(frameon=False)
    return fig


def greeks_vs_spot(S, K, T, r, sigma, option_type="call", q=0.0):
    spots = np.linspace(0.5 * K, 1.5 * K, 121)
    greeks = [bs_greeks(s, K, T, r, sigma, option_type, q) for s in spots]
    panels = [
        ("Delta", [g.delta for g in greeks]),
        ("Gamma", [g.gamma for g in greeks]),
        ("Vega (per 1% vol)", [g.vega / 100 for g in greeks]),
        ("Theta (per day)", [g.theta / 365 for g in greeks]),
    ]

    fig = Figure(figsize=(8, 5.5), layout="tight")
    axes = fig.subplots(2, 2, sharex=True)
    for i, (ax, (name, values)) in enumerate(zip(axes.flat, panels)):
        ax.plot(spots, values, color=BLUE, linewidth=2)
        ax.axvline(K, color=GREY, linewidth=1, linestyle=":")
        ax.axvline(S, color=ORANGE, linewidth=1.2)
        _style(ax, name, "Spot" if i >= 2 else "")
    fig.suptitle(f"{option_type.title()} greeks vs spot (orange = spot, dotted = strike)",
                 x=0.02, ha="left", fontsize=11)
    return fig
