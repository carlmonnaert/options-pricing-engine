import numpy as np

from ._common import check_inputs, payoff

def binomial_price(S, K, T, r, sigma, option_type="call", q=0.0, steps=500, american=False):
    """
    Calculate the price of an option using the binomial model.

    Parameters
    ----------
    S : float
        Current stock price.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility of the underlying asset.
    option_type : str, optional
        Type of the option, "call" or "put". Default is "call".
    q : float, optional
        Dividend yield. Default is 0.0.
    steps : int, optional
        Number of steps in the binomial tree. Default is 500.
    american : bool, optional
        Whether the option is American. Default is False.

    Returns
    -------
    float
        The option price.
    """
    option_type = check_inputs(S, K, T, r, sigma, option_type, q)
    if steps <= 0:
        raise ValueError("Number of steps must be a positive integer.")

    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    disc = np.exp(-r * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not 0 < p < 1:
        raise ValueError(f"p = {p:.4f} is not a probability, try more steps")

    j = np.arange(steps + 1)
    values = payoff(S * u ** (steps - 2 * j), K, option_type)

    for i in range(steps - 1, -1, -1):
        values = disc * (p * values[:-1] + (1 - p) * values[1:])
        if american:
            j = np.arange(i + 1)
            values = np.maximum(values, payoff(S * u ** (i - 2 * j), K, option_type))

    return float(values[0])