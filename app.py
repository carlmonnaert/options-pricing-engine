# streamlit run app.py
import time

import pandas as pd
import streamlit as st

from optionpricer import (
    binomial_price, bs_greeks, bs_price, fd_greeks, mc_price, put_call_parity_gap,
)
from optionpricer import plots

st.set_page_config(page_title="Option Pricing Engine", layout="wide")
st.title("Option Pricing Engine")
st.caption("Same option, three methods: Black-Scholes, a CRR binomial tree and Monte Carlo.")

# sidebar inputs
sb = st.sidebar
sb.header("Contract")
option_type = sb.radio("Type", ["call", "put"], horizontal=True)
S = sb.number_input("Spot S", min_value=0.01, value=100.0, step=1.0)
K = sb.number_input("Strike K", min_value=0.01, value=100.0, step=1.0)
T = sb.number_input("Maturity T (years)", min_value=0.01, value=1.0, step=0.25)

sb.header("Market")
sigma = sb.slider("Vol σ (%)", 1.0, 150.0, 20.0, 0.5) / 100
r = sb.slider("Rate r (%)", -2.0, 15.0, 5.0, 0.25) / 100
q = sb.slider("Dividend yield q (%)", 0.0, 10.0, 0.0, 0.25) / 100

sb.header("Numerics")
steps = sb.slider("Tree steps", 10, 2000, 500, 10)
n_path = sb.select_slider("MC paths", options=[1_000, 10_000, 50_000, 100_000, 500_000, 1_000_000],
                           value=100_000)
antithetic = sb.checkbox("Antithetic variates", value=True)
control_variate = sb.checkbox("Control variate (S_T)", value=True)
seed = int(sb.number_input("Seed", min_value=0, value=42, step=1))

params = dict(S=S, K=K, T=T, r=r, sigma=sigma, option_type=option_type, q=q)
mc_settings = dict(n_path=n_path, antithetic=antithetic,
                   control_variate=control_variate, seed=seed)

try:
    t0 = time.perf_counter()
    bs = bs_price(**params)
    t_bs = time.perf_counter() - t0

    t0 = time.perf_counter()
    tree_eu = binomial_price(**params, steps=steps)
    tree_am = binomial_price(**params, steps=steps, american=True)
    t_tree = time.perf_counter() - t0

    t0 = time.perf_counter()
    mc = mc_price(**params, **mc_settings)
    t_mc = time.perf_counter() - t0
except ValueError as e:
    st.error(str(e))
    st.stop()

# prices
c1, c2, c3, c4 = st.columns(4)
c1.metric("Black-Scholes", f"{bs:.4f}")
c2.metric("Tree (European)", f"{tree_eu:.4f}", delta=f"{tree_eu - bs:+.4f} vs BS", delta_color="off")
c3.metric("Monte Carlo", f"{mc.price:.4f}", delta=f"{mc.price - bs:+.4f} vs BS", delta_color="off")
c4.metric("Tree (American)", f"{tree_am:.4f}",
          delta=f"early exercise {tree_am - tree_eu:+.4f}", delta_color="off")

lo, hi = mc.ci95
where = "inside" if lo <= bs <= hi else "outside (should happen ~1 time in 20)"
st.write(f"MC 95% CI: [{lo:.4f}, {hi:.4f}], std error {mc.std_error:.5f} with "
         f"{mc.n_path:,} paths. BS price is {where}.")
parity = put_call_parity_gap(S, K, T, r, sigma, q)
st.write(f"Put-call parity gap: {parity:.1e}. Time: BS {t_bs * 1e3:.2f} ms, "
         f"tree {t_tree * 1e3:.1f} ms, MC {t_mc * 1e3:.1f} ms.")

# greeks
st.subheader("Greeks")
exact = bs_greeks(**params).as_dict()
fd_bs = fd_greeks(**params).as_dict()
fd_mc = fd_greeks(**params, pricer=mc_price, rel_bump=1e-2, **mc_settings).as_dict()

# show vega/rho per 1% and theta per day
rows = [("Delta", "delta", 1), ("Gamma", "gamma", 1), ("Vega (per 1%)", "vega", 0.01),
        ("Theta (per day)", "theta", 1 / 365), ("Rho (per 1%)", "rho", 0.01)]
table = pd.DataFrame(
    [(name, exact[k] * s, fd_bs[k] * s, fd_mc[k] * s) for name, k, s in rows],
    columns=["Greek", "Analytic", "Finite diff (BS)", "Finite diff (MC, same seed)"],
)
st.dataframe(table.style.format(precision=5, subset=table.columns[1:]),
             hide_index=True, use_container_width=True)
st.caption("MC gamma is noisy: second differences of a kinked payoff blow up the simulation noise.")


@st.cache_data
def mc_fig(model, seed, antithetic, control_variate):
    return plots.mc_convergence(*model, seed=seed, antithetic=antithetic,
                                control_variate=control_variate)


@st.cache_data
def tree_fig(model):
    return plots.binomial_convergence(*model)


model = (S, K, T, r, sigma, option_type, q)
tab1, tab2, tab3, tab4 = st.tabs(["MC convergence", "Tree convergence", "Value vs spot", "Greeks vs spot"])
with tab1:
    st.pyplot(mc_fig(model, seed, antithetic, control_variate))
with tab2:
    st.pyplot(tree_fig(model))
with tab3:
    st.pyplot(plots.price_vs_spot(*model))
with tab4:
    st.pyplot(plots.greeks_vs_spot(*model))

with st.expander("Formulas"):
    st.latex(r"C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2), \quad "
             r"d_{1,2} = \frac{\ln(S/K) + (r - q \pm \tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}")
    st.latex(r"u = e^{\sigma\sqrt{\Delta t}},\; d = 1/u,\; p = \frac{e^{(r-q)\Delta t} - d}{u - d}")
    st.latex(r"S_T = S \exp\left((r - q - \tfrac{1}{2}\sigma^2)T + \sigma\sqrt{T}Z\right)")
