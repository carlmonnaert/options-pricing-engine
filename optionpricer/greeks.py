from .black_scholes import Greeks, bs_price

def fd_greeks(S, K, T, r, sigma, option_type="call", q=0.0,
              pricer=bs_price, rel_bump=1e-3, **pricer_kwargs):
    """
    Compute the option Greeks using finite difference method.

    Parameters:
    S : float
        Spot price of the underlying asset.
    K : float
        Strike price of the option.
    T : float
        Time to maturity (in years).
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility of the underlying asset.
    option_type : str, optional
        Type of the option ("call" or "put"). Default is "call".
    q : float, optional
        Dividend yield. Default is 0.0.
    pricer : callable, optional
        Function to compute the option price. Default is bs_price.
    rel_bump : float, optional
        Relative bump size for finite difference. Default is 1e-3.
    **pricer_kwargs : dict
        Additional keyword arguments for the pricer function.

    Returns:
    Greeks
        Named tuple containing delta, gamma, vega, rho, and theta.
    """
    def price(S_=S, T_=T, r_=r, sigma_=sigma):
        res = pricer(S_, K, T_, r_, sigma_, option_type, q, **pricer_kwargs)
        return float(getattr(res, "price", res))

    dS = rel_bump * S
    dv = rel_bump * sigma
    dr = 1e-4
    dT = min(rel_bump * T, 0.5 * T)

    p0 = price()
    p_up = price(S_=S + dS)
    p_dn = price(S_=S - dS)
    
    delta = (p_up - p_dn) / (2 * dS)
    gamma = (p_up - 2 * p0 + p_dn) / (dS ** 2)
    vega = (price(sigma_=sigma + dv) - price(sigma_=sigma - dv)) / (2 * dv)
    rho = (price(r_=r + dr) - price(r_=r - dr)) / (2 * dr)
    theta = -(price(T_=T + dT) - price(T_=T - dT)) / (2 * dT)

    return Greeks(delta=delta, gamma=gamma, vega=vega, rho=rho, theta=theta)