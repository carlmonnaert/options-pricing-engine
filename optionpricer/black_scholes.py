"""
Black-Scholes option pricing model implementation, with continuous divident yield q.
Rates and vols are decimals (e.g., 0.05 for 5%) and T is in years.
"""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import norm

from ._common import check_inputs

def _d1_d2(S, K, T, r, sigma, q=0.0):
    """
    Calculate the d1 and d2 terms used in the Black-Scholes formula.

    Parameters
    ----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to maturity in years
    r : float
        Risk-free interest rate
    sigma : float
        Volatility of the underlying asset
    q : float, optional
        Continuous dividend yield (default is 0.0)

    Returns
    -------
    d1 : float
    d2 : float
    """
    d1 = (np.log(S/K) + (r - q + sigma**2 / 2) * T ) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

def bs_price(S, K, T, r, sigma, option_type="call", q=0.0):
    """
    Calculate the Black-Scholes price for a European call or put option.

    Parameters
    ----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to maturity in years
    r : float
        Risk-free interest rate
    sigma : float
        Volatility of the underlying asset
    q : float, optional
        Continuous dividend yield (default is 0.0)
    option_type : str, optional
        Type of the option, either 'call' or 'put' (default is 'call')

    Returns
    -------
    price : float
        Black-Scholes price of the option
    """
    option_type = check_inputs(S, K, T, r, sigma, option_type, q)
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Must be 'call' or 'put'.")

@dataclass(frozen=True)
class Greeks:
    """
    Dataclass representing the Greeks of an option.
    """
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

    def as_dict(self):
        return asdict(self)

def bs_greeks(S, K, T, r, sigma, option_type="call", q=0.0):
    """
    Calculate the Greeks for a European call or put option using the Black-Scholes model.

    Parameters
    ----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to maturity in years
    r : float
        Risk-free interest rate
    sigma : float
        Volatility of the underlying asset
    option_type : str, optional
        Type of the option, either 'call' or 'put' (default is 'call')
    q : float, optional
        Continuous dividend yield (default is 0.0)

    Returns
    -------
    Greeks
        Dataclass containing the option Greeks: delta, gamma, vega, theta, rho
    """
    option_type = check_inputs(S, K, T, r, sigma, option_type, q)
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    sqT = np.sqrt(T)
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)
    nd1 = norm.pdf(d1)

    gamma = disc_q * nd1 / (S * sigma * sqT)
    vega = S * disc_q * nd1 * sqT
    theta = - S * disc_q * nd1 * sigma / (2 * sqT)

    if option_type == "call":
        delta = disc_q * norm.cdf(d1)
        theta += -r * K * disc_r * norm.cdf(d2) + q * S * disc_q * norm.cdf(d1)
        rho = K * T * disc_r * norm.cdf(d2)
    else:
        delta = disc_q * (norm.cdf(d1) - 1)
        theta += r * K * disc_r * norm.cdf(-d2) - q * S * disc_q * norm.cdf(-d1)
        rho = -K * T * disc_r * norm.cdf(-d2)

    return Greeks(float(delta), float(gamma), float(vega), float(theta), float(rho))

def put_call_parity_gap(S, K, T, r, sigma, q=0.0, pricer=bs_price, **kwargs):
    call = pricer(S, K, T, r, sigma, option_type="call", q=q, **kwargs)
    put = pricer(S, K, T, r, sigma, option_type="put", q=q, **kwargs)
    return float(call - put - S * np.exp(-q * T) + K * np.exp(-r * T))