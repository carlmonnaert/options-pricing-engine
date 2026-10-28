# regenerates the plots in docs/ used by the README
# run from the repo root: python scripts/make_figures.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from optionpricer import plots

params = (100, 100, 1.0, 0.05, 0.20, "call", 0.0)
out = ROOT / "docs"
out.mkdir(exist_ok=True)

plots.mc_variance_reduction(*params, seed=42).savefig(out / "mc_variance_reduction.png", dpi=150)
plots.binomial_convergence(*params).savefig(out / "binomial_convergence.png", dpi=150)
plots.greeks_vs_spot(*params).savefig(out / "greeks_vs_spot.png", dpi=150)
print("saved plots to", out)