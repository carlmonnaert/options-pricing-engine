"""
Monte Carlo simulation for option pricing.

Under GMB assumption, we can sample S_T directly, and do not need to simulate the entire path:

S_T = S * exp((r - q - sigma**2 / 2) * T + sigma * sqrt(T) * Z)
where Z ~ N(0, 1)
"""
from dataclasses import dataclass

import numpy as np

from ._common import check_inputs, payoff

@dataclass(frozen=True)
class MCResult:
    price: float
    std_error: float
    n_path: int

    @property
    def ci95(self):
        return self.price - 1.96 * self.std_error, self.price + 1.96 * self.std_error


def mc_price(S, K, T, r, sigma, option_type="call", q=0.0, n_path=100_000,
             antithetic=True, control_variate=False, seed=None):
    """
    Monte Carlo pricing of European options under the Geometric Brownian Motion assumption.

    Parameters
    ----------
    S : float
        Spot price of the underlying asset.
    K : float
        Strike price of the option.
    T : float
        Time to maturity in years.
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility of the underlying asset.
    option_type : str, optional
        "call" or "put", by default "call".
    q : float, optional
        Dividend yield, by default 0.0.
    n_path : int, optional
        Number of Monte Carlo paths, by default 100_000.
    antithetic : bool, optional
        Use antithetic variates for variance reduction, by default True.
    control_variate : bool, optional
        Use control variate technique for variance reduction, by default False.
    seed : int, optional
        Random seed for reproducibility, by default None.

    Returns
    -------
    MCResult
        Dataclass containing the option price, standard error, and number of paths used.
    """
    option_type = check_inputs(S, K, T, r, sigma, option_type, q)
    if n_path <= 1:
        raise ValueError("n_path must be greater than 1")

    rng = np.random.default_rng(seed)
    n = max(n_path//2, 2) if antithetic else n_path
    Z = rng.standard_normal(n)

    drift = (r - q - sigma**2 / 2) * T
    vol = sigma * np.sqrt(T)
    disc = np.exp(-r * T)

    def simulate(z):
        ST = S * np.exp(drift + vol * z)
        return disc * payoff(ST, K, option_type), disc * ST

    Y, X = simulate(Z)
    if antithetic:
        Y_2, X_2 = simulate(-Z)
        Y = 0.5 * (Y + Y_2)
        X = 0.5 * (X + X_2)

    if control_variate:
        cov = np.cov(Y, X)
        beta = cov[0, 1] / cov[0, 0] if cov[0, 0] > 0 else 0.0
        Y = Y - beta * (X - S * np.exp(-q * T))

    price = Y.mean()
    std_error = Y.std(ddof=1) / np.sqrt(len(Y))
    return MCResult(float(price), float(std_error), 2 * n if antithetic else n)