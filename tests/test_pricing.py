import numpy as np
import pytest

from optionpricer import (
    binomial_price, bs_greeks, bs_price, fd_greeks, mc_price, put_call_parity_gap,
)

# S, K, T, r, sigma, q
CASES = [
    (100, 100, 1.0, 0.05, 0.20, 0.00),  # ATM
    (100, 120, 0.5, 0.03, 0.30, 0.01),  # OTM call
    (100, 80, 2.0, 0.01, 0.15, 0.02),   # ITM call, longer maturity
    (50, 55, 0.25, 0.00, 0.45, 0.00),   # zero rate, high vol
]
TYPES = ["call", "put"]


def test_bs_hull_example():
    # standard textbook example (Hull): S=K=100, T=1, r=5%, vol=20%
    assert bs_price(100, 100, 1, 0.05, 0.2, "call") == pytest.approx(10.4506, abs=1e-4)
    assert bs_price(100, 100, 1, 0.05, 0.2, "put") == pytest.approx(5.5735, abs=1e-4)


@pytest.mark.parametrize("case", CASES)
def test_bs_put_call_parity(case):
    assert put_call_parity_gap(*case) == pytest.approx(0, abs=1e-10)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("opt", TYPES)
def test_bs_no_arbitrage_bounds(case, opt):
    S, K, T, r, sigma, q = case
    price = bs_price(S, K, T, r, sigma, opt, q)
    fwd_S = S * np.exp(-q * T)
    pv_K = K * np.exp(-r * T)
    if opt == "call":
        assert max(fwd_S - pv_K, 0) - 1e-12 <= price <= fwd_S + 1e-12
    else:
        assert max(pv_K - fwd_S, 0) - 1e-12 <= price <= pv_K + 1e-12


@pytest.mark.parametrize("bad", [
    {"S": -1}, {"K": 0}, {"T": 0}, {"sigma": 0}, {"option_type": "straddle"},
])
def test_bad_inputs(bad):
    args = {"S": 100, "K": 100, "T": 1, "r": 0.05, "sigma": 0.2, "option_type": "call"}
    args.update(bad)
    with pytest.raises(ValueError):
        bs_price(**args)

@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("opt", TYPES)
def test_greeks_vs_finite_differences(case, opt):
    exact = bs_greeks(*case[:5], option_type=opt, q=case[5]).as_dict()
    approx = fd_greeks(*case[:5], option_type=opt, q=case[5]).as_dict()
    for name in exact:
        assert approx[name] == pytest.approx(exact[name], rel=1e-4, abs=1e-6), name


def test_greek_signs():
    call = bs_greeks(100, 100, 1, 0.05, 0.2, "call")
    put = bs_greeks(100, 100, 1, 0.05, 0.2, "put")
    assert 0 < call.delta < 1
    assert -1 < put.delta < 0
    assert call.delta - put.delta == pytest.approx(1)  # since q = 0
    assert call.gamma == pytest.approx(put.gamma)
    assert call.vega == pytest.approx(put.vega)
    assert call.rho > 0 > put.rho


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("opt", TYPES)
def test_tree_converges_to_bs(case, opt):
    exact = bs_price(*case[:5], option_type=opt, q=case[5])
    # CRR oscillates between odd and even N, averaging N and N+1 smooths it out
    tree = (binomial_price(*case[:5], option_type=opt, q=case[5], steps=1000)
            + binomial_price(*case[:5], option_type=opt, q=case[5], steps=1001)) / 2
    assert tree == pytest.approx(exact, abs=5e-3)


def test_tree_error_goes_down():
    exact = bs_price(100, 100, 1, 0.05, 0.2, option_type="call", q=0.0)
    errs = [abs(binomial_price(100, 100, 1, 0.05, 0.2, option_type="call", q=0.0, steps=n) - exact)
            for n in (10, 100, 1000)]
    assert errs[0] > errs[1] > errs[2]


def test_american_put_worth_more():
    eu = binomial_price(100, 110, 1, 0.08, 0.2, option_type="put", steps=800)
    am = binomial_price(100, 110, 1, 0.08, 0.2, option_type="put", steps=800, american=True)
    assert am > eu + 0.1
    assert am >= 10  # at least intrinsic value


def test_american_call_no_dividend_same_as_european():
    eu = binomial_price(100, 100, 1, 0.05, 0.2, option_type="call", steps=500)
    am = binomial_price(100, 100, 1, 0.05, 0.2, option_type="call", steps=500, american=True)
    assert am == pytest.approx(eu, abs=1e-10)


def test_tree_bad_probability():
    # huge rate, tiny vol, 1 step -> p > 1
    with pytest.raises(ValueError):
        binomial_price(100, 100, 1, 0.9, 0.05, option_type="call", steps=1)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("opt", TYPES)
def test_mc_close_to_bs(case, opt):
    exact = bs_price(*case[:5], option_type=opt, q=case[5])
    res = mc_price(*case[:5], option_type=opt, q=case[5], n_path=200_000, seed=42)
    assert abs(res.price - exact) < 4 * res.std_error


def test_variance_reduction_helps():
    args = (100, 100, 1, 0.05, 0.2, "call")
    plain = mc_price(*args, n_path=100_000, antithetic=False, seed=1)
    anti = mc_price(*args, n_path=100_000, antithetic=True, seed=1)
    both = mc_price(*args, n_path=100_000, antithetic=True, control_variate=True, seed=1)
    assert both.std_error < anti.std_error < plain.std_error


def test_mc_seed_reproducible():
    a = mc_price(100, 100, 1, 0.05, 0.2, "put", seed=7)
    b = mc_price(100, 100, 1, 0.05, 0.2, "put", seed=7)
    assert a.price == b.price


def test_mc_put_call_parity():
    gap = put_call_parity_gap(100, 100, 1, 0.05, 0.2, pricer=mc_price, n_path=200_000, seed=3)
    assert abs(gap) < 0.05


def test_mc_delta_same_seed():
    exact = bs_greeks(100, 100, 1, 0.05, 0.2, "call").delta
    mc = fd_greeks(100, 100, 1, 0.05, 0.2, "call", pricer=mc_price, rel_bump=1e-2,
                   n_path=200_000, seed=11)
    assert mc.delta == pytest.approx(exact, abs=0.01)
