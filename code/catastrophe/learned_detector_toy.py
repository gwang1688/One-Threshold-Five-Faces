"""
learned_detector_toy.py  --  A first step beyond the analytic surrogate.

The catastrophe paper imposes the detector A as a logistic of prescribed steepness.
Here the detector is LEARNED: a logistic classifier trained to separate "loyal"
from "hacking" footprints in a d-dim feature space, so its steepness is EMERGENT
(set by the class separability). A scalar policy ascends the payoff
    w * (1 - A_hat(rho)) - 1/2 c rho^2
by gradient on the learned score A_hat (queried by sampling footprints). Ramping w
up then down, we measure hysteresis: it is negligible at low separability and grows
as the learned detector sharpens -- the same law as Theorem 1 for the imposed logistic.

This is a controlled step beyond the surrogate (learned + sampled detector, real
gradient), still far from a deployed agent. CPU-only, numpy only.
"""
import numpy as np


def make_detector(d, sep, n=4000, seed=0):
    """Train a logistic detector; separability `sep` sets the emergent steepness."""
    rng = np.random.default_rng(seed)
    Xl = rng.normal(0, 1, (n, d)); Xl[:, 0] -= sep
    Xh = rng.normal(0, 1, (n, d)); Xh[:, 0] += sep
    X = np.vstack([Xl, Xh]); y = np.r_[np.zeros(n), np.ones(n)]
    wv = np.zeros(d); b = 0.0
    for _ in range(300):
        p = 1/(1 + np.exp(-(X @ wv + b)))
        g = p - y
        wv -= 0.05*(X.T @ g)/len(y) + 1e-3*wv
        b -= 0.05*g.mean()
    return wv, b


def footprint(rho, d, sep, rng):
    """A footprint whose hidden-ness is rho: loyal(-sep) -> hacking(+sep) along e1."""
    x = rng.normal(0, 1, d)
    x[0] = (2*rho - 1)*sep
    return x


def Ahat(rho, d, sep, wv, b, rng, m=64):
    """Learned detection prob of LOYALTY (high ~ caught) for a rho-footprint."""
    xs = np.array([footprint(rho, d, sep, rng) for _ in range(m)])
    return 1 - (1/(1 + np.exp(-(xs @ wv + b)))).mean()


def ramp(sep, d=8, c=1.0, mu=0.02, steps=60, samples=40, seed=0):
    """Ramp w up then down; return (w, up-branch rho, down-branch rho)."""
    rng = np.random.default_rng(seed)
    wv, b = make_detector(d, sep, seed=seed)

    def eq(w, rho):
        for _ in range(samples):
            e = 0.03
            dA = (Ahat(min(rho+e, 1), d, sep, wv, b, rng)
                  - Ahat(max(rho-e, 0), d, sep, wv, b, rng)) / (2*e)
            rho = np.clip(rho + mu*(w*(-dA) - c*rho), 0, 1)  # ascend payoff (h = -dA)
        return rho

    ws = np.linspace(0.2, 3.0, steps)
    rho = 0.02
    up = [rho := eq(w, rho) for w in ws]
    dn = [rho := eq(w, rho) for w in ws[::-1]]
    return ws, np.array(up), np.array(dn[::-1])


def main():
    print("Learned-detector experiment: hysteresis vs detector separability (emergent steepness).")
    print("Ramp w up then down; hysteresis = max gap between the branches.\n")
    print(f"{'separability':>13} {'max hysteresis':>15}")
    for sep in (0.6, 1.0, 1.4, 2.0, 2.5, 3.0):
        hy = np.mean([np.max(np.abs(ramp(sep, seed=s)[1] - ramp(sep, seed=s)[2]))
                      for s in range(3)])
        print(f"{sep:>13.1f} {hy:>15.3f}")
    print("\nExpected: hysteresis ~0 at low separability, growing to ~0.6 as the learned")
    print("detector sharpens -- the fold survives a learned detector (Theorem 1).")


if __name__ == "__main__":
    main()
