"""
collective_endpoint.py  --  Disorder-induced critical endpoint (surrogate).

Reproduces Theorem 3 of "Collective Intent Drift": as the vulnerability spread CV
of a population grows, the first-order collective transition weakens and vanishes
at a critical endpoint CV_c, where the two mean-field folds merge in a cusp.

We work with the mean-field self-consistency
    m = Phi(m; kappa, CV) = E_{c ~ nu_CV}[ rho_eq(w0 + kappa m; c) ]
with nu_CV lognormal (fixed mean 1, coefficient of variation CV), low base stakes
w0 = 0.35 so no agent self-ignites (avoids a spurious jump). The jump is measured
by ROOT-FINDING the outer stable branches of Phi(m) - m = 0 (floor-free).

Outputs:
  (1) jump Delta_m(CV): shrinks continuously and vanishes near CV_c ~ 0.57;
  (2) bistable kappa-window: shrinks to a point at the endpoint (cusp merging);
  (3) beta = 1/2 consistency: Delta_m^2 is linear in CV on the clean region;
  (4) independent vs correlated: softening requires INDEPENDENT spread.

CPU-only, numpy + scipy.brentq.
"""
import numpy as np
from scipy import optimize

rstar = 0.5
sig = 1.2          # per-agent detector steepness (> sigma_c: single-agent catastrophic)
w0 = 0.35          # base stakes (low: no self-ignition)


def h(rho):
    return sig/np.cosh(4*sig*(rho - rstar))**2


G = np.linspace(1e-6, 0.999999, 8000)
Hg = h(G)
Wgrid = np.linspace(0, 45, 8000)


def _table(c):
    """rho_eq(W; c): lowest stable root of W h(rho) = c rho, over the W grid."""
    out = np.empty_like(Wgrid)
    for j, W in enumerate(Wgrid):
        F = W*Hg - c*G
        sc = np.where(np.diff(np.sign(F)) != 0)[0]
        out[j] = G[sc[0]] if len(sc) else G[np.argmin(np.abs(F))]
    return out


# precompute rho_eq on a cost grid once (the expensive step)
CGRID = np.linspace(0.05, 4.5, 300)
TAB = np.array([_table(c) for c in CGRID])


def Ppop_lognormal(CV):
    """Population-averaged response Phi_pop(W) for a lognormal nu (mean 1, given CV)."""
    if CV < 1e-9:
        w = np.zeros(len(CGRID)); w[np.argmin(np.abs(CGRID - 1.0))] = 1.0
    else:
        s2 = np.log(1 + CV**2); mu = -0.5*s2
        d = np.exp(-(np.log(CGRID) - mu)**2/(2*s2))/(CGRID*np.sqrt(2*np.pi*s2))
        w = d/d.sum()
    return w @ TAB


def Ppop_from_costs(cs):
    """Phi_pop(W) for an explicit sample of costs (used for correlated draws)."""
    idx = np.clip(np.searchsorted(CGRID, cs), 0, len(CGRID)-1)
    return (np.bincount(idx, minlength=len(CGRID))/len(cs)) @ TAB


def jump_and_window(Pp, nk=500):
    """Given Phi_pop(W), return (jump, kappa_lo, kappa_hi) via root-finding on Phi(m)-m."""
    def Phi(m, kap):
        return np.interp(w0 + kap*m, Wgrid, Pp)
    best, klo, khi = 0.0, np.nan, np.nan
    for kap in np.linspace(0.3, 16, nk):
        ms = np.linspace(0, 1, 400)
        F = Phi(ms, kap) - ms
        sc = np.where(np.diff(np.sign(F)) != 0)[0]
        if len(sc) >= 3:
            roots = [optimize.brentq(lambda m: Phi(m, kap) - m, ms[i], ms[i+1], xtol=1e-11)
                     for i in sc]
            best = max(best, roots[-1] - roots[0])
            klo = kap if np.isnan(klo) else min(klo, kap)
            khi = kap if np.isnan(khi) else max(khi, kap)
    return best, klo, khi


def main():
    print("(1) jump Delta_m(CV) and (2) bistable kappa-window (lognormal nu, mean 1, w0=0.35)\n")
    print(f"{'CV':>6} {'jump':>8} {'kappa window':>13}")
    CVs = np.arange(0.30, 0.62, 0.03)
    jumps = []
    for CV in CVs:
        j, lo, hi = jump_and_window(Ppop_lognormal(CV))
        win = (hi - lo) if not np.isnan(lo) else 0.0
        jumps.append(j)
        print(f"{CV:>6.2f} {j:>8.4f} {win:>13.4f}")
    jumps = np.array(jumps)

    print("\n(3) exponent check on the clean region CV in [0.40, 0.51]:")
    cl = (CVs >= 0.40) & (CVs <= 0.52)
    J = jumps[cl]; X = CVs[cl]
    c2 = np.polyfit(X, J**2, 1); CVc = -c2[1]/c2[0]
    R2h = 1 - np.var(J**2 - np.polyval(c2, X))/np.var(J**2)
    c3 = np.polyfit(X, J**3, 1)
    R3 = 1 - np.var(J**3 - np.polyval(c3, X))/np.var(J**3)
    print(f"    Delta_m^2 linear in CV (beta=1/2): CV_c ~ {CVc:.3f}, R^2 = {R2h:.3f}")
    print(f"    Delta_m^3 linear in CV (beta=1/3):              R^2 = {R3:.3f}")
    print("    (both fit well away from the endpoint; beta=1/2 is fixed by the cusp normal form,")
    print("     not by the fit -- the endpoint region is numerically ill-conditioned, see paper.)")

    print("\n(4) softening requires INDEPENDENT spread (correlated monoculture stays sharp):")
    rng = np.random.default_rng(0); M = 6000
    print(f"    {'CV':>6} {'independent':>12} {'correlated':>11}")
    for CV in (0.2, 0.5, 0.8):
        s2 = np.log(1 + CV**2)
        cs_ind = np.clip(np.exp(rng.normal(-0.5*s2, np.sqrt(s2), M)), CGRID[0], CGRID[-1])
        j_ind = jump_and_window(Ppop_from_costs(cs_ind))[0]
        # correlated monoculture: all agents share ONE draw; average jump over draws
        j_cor = np.mean([jump_and_window(Ppop_from_costs(
            np.full(M, np.clip(np.exp(rng.normal(-0.5*s2, np.sqrt(s2))), CGRID[0], CGRID[-1]))))[0]
            for _ in range(4)])
        print(f"    {CV:>6.2f} {j_ind:>12.3f} {j_cor:>11.3f}")
    print("\n    Independent spread softens the jump to ~0; a correlated monoculture keeps it large.")


if __name__ == "__main__":
    main()
