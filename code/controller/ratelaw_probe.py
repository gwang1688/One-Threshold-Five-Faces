"""
ratelaw_probe.py  --  The rate law (rate-induced tipping) of the closed loop.

Reproduces the rate-law experiment of the controller paper (Theorem 4):
the closed loop succeeds iff the actuator engages before the (noise-induced)
tip, i.e. wdot * tau_act < L(sigma) * w_+. Sweeping the actuator delay tau_act
at fixed ramp rate, the loyal-success rate holds up to a critical delay and then
collapses (rate-induced tipping); the critical delay shrinks as sigma grows
(smaller CSD lead L(sigma)).

CPU-only, numpy only.
"""
import numpy as np

rstar = 0.5


def A(rho, sig):
    return 0.5 + 0.5/(1 + np.exp(8*sig*(rho - rstar)))


def h(rho, sig):
    return sig/np.cosh(4*sig*(rho - rstar))**2


def wplus(sig, c=1.0):
    g = np.linspace(1e-4, rstar-1e-4, 200000)
    return (c*g/h(g, sig)).max()


def rho_plus(sig, c=1.0):
    g = np.linspace(1e-4, rstar-1e-4, 200000)
    return g[np.argmax(c*g/h(g, sig))]


def run(sig, tau_act, c=1.0, mu=0.03, xi=0.012, seed=0, steps=16000,
        win=250, cap_frac=0.9, warmup=3000, var_trigger=3e-4):
    """Closed loop with an actuator delay tau_act between trigger and engagement.
    Returns final rho (loyal if < 0.3)."""
    wp = wplus(sig, c)
    rho_cap = cap_frac*rho_plus(sig, c)
    ws = np.linspace(0.3*wp, 1.5*wp, steps)
    rng = np.random.default_rng(seed)
    rho = 0.02
    pend = None        # scheduled engagement step
    engaged = False
    buf = []
    for t in range(steps):
        w = ws[t]
        buf.append(rho)
        if len(buf) > win:
            buf.pop(0)
        if not engaged and pend is None and len(buf) == win and t > warmup:
            seg = np.array(buf)
            seg = seg - np.polyval(np.polyfit(np.arange(win), seg, 1), np.arange(win))
            if seg.var() > var_trigger:
                pend = t + int(tau_act)        # engage only after the actuator delay
        if pend is not None and t >= pend:
            engaged = True
        drift = mu * (w*h(rho, sig) - c*rho)
        rho = rho + drift + xi*np.sqrt(mu)*rng.standard_normal()
        hi = rho_cap if engaged else 1.0
        rho = min(hi, max(0.0, rho))
    return rho


def main():
    taus = (0, 500, 1000, 2000, 3000, 4000, 6000, 8000)
    print("Rate law: loyal-success rate vs actuator delay tau_act (6 seeds).")
    print("Success holds up to a critical delay, then collapses (rate-induced tipping).\n")
    print(f"{'tau_act':>8} " + " ".join(f"{'sig='+str(s):>9}" for s in (1.2, 1.4, 2.0)))
    for tau in taus:
        row = [f"{tau:>8}"]
        for sig in (1.2, 1.4, 2.0):
            succ = np.mean([run(sig, tau, seed=s) < 0.3 for s in range(6)])
            row.append(f"{succ:>9.2f}")
        print(" ".join(row))
    print("\nExpected: ~100% success at small tau_act; collapse past a critical delay;")
    print("          the critical delay shrinks as sigma grows (smaller CSD lead L(sigma)).")


if __name__ == "__main__":
    main()
