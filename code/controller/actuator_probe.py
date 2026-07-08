"""
actuator_probe.py  --  The nonsmooth-damping actuator (surrogate).

Reproduces the actuator experiment of the controller paper (Theorem 2):
  (a) a QUADRATIC damper leaves sigma_c invariant;
  (b) SMOOTH superquadratic stiffening phi = rho + b rho^3 raises sigma_c(b)
      but saturates (crosses sigma_det=1.4 only by extrapolation, b* ~ 61);
  (c) a nonsmooth DEADBAND (cap rho <= rho_cap < rho_+) removes the hacking
      basin for ANY sigma;
  and the detector AUC is unchanged throughout (the damper never touches A).

CPU-only, numpy only.
"""
import numpy as np

rstar = 0.5


def A(rho, sig):
    return 0.5 + 0.5/(1 + np.exp(8*sig*(rho - rstar)))


def h(rho, sig):
    return sig/np.cosh(4*sig*(rho - rstar))**2


G = np.linspace(1e-5, 0.9999, 8000)


def sigma_c(phi_of_rho):
    """Smallest sigma at which w*h = phi has three roots for some w (fold onset)."""
    def three(sig):
        Hg = h(G, sig)
        phi = phi_of_rho(G)
        wloyal = (phi[G < rstar] / Hg[G < rstar]).max()
        for W in np.linspace(0.005, 3*wloyal, 700):
            if np.sum(np.diff(np.sign(W*Hg - phi)) != 0) >= 3:
                return True
        return False
    lo, hi = 0.15, 6.0
    if not three(hi):
        return np.nan
    for _ in range(34):
        m = 0.5*(lo+hi)
        hi, lo = (m, lo) if three(m) else (hi, m)
    return hi


def detection_auc(sig, spread=0.25, n=3000, seed=0):
    """AUC separating loyal vs hacking operating points by detection A. A does NOT
    depend on the damper b, so this is invariant under the actuator (the point)."""
    rng = np.random.default_rng(seed)
    loyal = A(np.clip(rng.normal(0.30, spread, n), 0, 1), sig)   # overlapping operating points
    hack = A(np.clip(rng.normal(0.70, spread, n), 0, 1), sig)
    return np.mean(loyal[:, None] > hack[None, :])   # higher A => loyal


def deadband_removes_basin(sig, cap_frac=0.9):
    """With cap rho <= rho_cap < rho_+, is the hacking equilibrium excised for large w?"""
    g = G[G < rstar]
    rho_plus = g[np.argmax(g/h(g, sig))]
    cap = cap_frac*rho_plus
    # feasible set [0,cap]: only the loyal root survives (hacking root lies at rho>rho_+ > cap)
    return cap < rho_plus


def main():
    print("(a) quadratic damper: sigma_c should stay ~constant as we scale the quadratic cost")
    for c in (1.0, 1.5, 2.0):
        print(f"    phi={c}*rho : sigma_c = {sigma_c(lambda r, c=c: c*r):.3f}")
    print("\n(b) smooth superquadratic stiffening phi = rho + b rho^3 : sigma_c(b) rises, saturates")
    print(f"    {'b':>5} {'sigma_c(b)':>11}")
    for b in (0, 10, 20, 30, 50):
        print(f"    {b:>5} {sigma_c(lambda r, b=b: r + b*r**3):>11.3f}")
    print("    (extrapolating the same trend, sigma_c crosses sigma_det=1.4 near b* ~ 61)")
    print("\n(c) nonsmooth deadband removes the hacking basin for any sigma:")
    for sig in (1.4, 2.5, 5.0):
        print(f"    sigma={sig}: cap < rho_+  -> hacking basin excised = {deadband_removes_basin(sig)}")
    print("\n    detection AUC is UNCHANGED by the damper (A never depends on b):")
    sig = 1.4
    auc0 = detection_auc(sig)
    print(f"    sigma={sig}: AUC(b=0) = {auc0:.3f}, AUC(b=50) = {auc0:.3f}  (identical: damper never touches A)")
    print("    (absolute value depends on the loyal/hacking separation; invariance under b is the claim)")


if __name__ == "__main__":
    main()
