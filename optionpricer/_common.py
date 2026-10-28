import numpy as np

def check_inputs(S, K, T, r, sigma, option_type, q=0.0):
    """
    Checks the validity of the input parameters for option pricing.
    Parameters:
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to maturity (in years)
    r : float
        Risk-free interest rate
    sigma : float
        Volatility of the underlying asset
    option_type : str
        Type of the option, either 'call' or 'put'
    q : float, optional
        Dividend yield (default is 0.0)

    Returns:
    str
        The validated option type in lowercase
    """
    option_type = option_type.lower()
    if option_type not in ["call", "put"]:
        raise ValueError("option_type must be 'call' or 'put'")
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive")
    if T <= 0:
        raise ValueError("T must be positive (in years)")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if not np.all(np.isfinite([S, K, T, r, sigma, q])):
        raise ValueError("All inputs must be finite numbers")
    return option_type

def payoff(ST, K, option_type):
    """
    Calculates the payoff of an option at maturity.

    Parameters:
    ST : float
        Stock price at maturity
    K : float
        Strike price
    option_type : str
        Type of the option, either 'call' or 'put'

    Returns:
    float
        The payoff of the option
    Notes: Assumes that option_type has been validated and is either 'call' or 'put'
    """
    if option_type == "call":
        return max(ST - K, 0)
    elif option_type == "put":
        return max(K - ST, 0)
    else:
        raise ValueError("Invalid option_type, must be 'call' or 'put'")