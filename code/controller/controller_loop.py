"""
controller_loop.py  --  Closed-loop controller for reward hacking (surrogate).

Reproduces the central closed-loop experiment of
"A Closed-Loop Controller for Reward Hacking: Critical Slowing Down and Nonsmooth Damping".

Plant (noisy best-response flow on the evasion coordinate rho in [0,1]):
    d rho = mu * dPi/drho dt + xi dW,   Pi = w (1 - A(rho)) - C(rho)
with logistic detector A of steepness sigma, quadratic base cost C = 1/2 rho^2.

Controller:
  * SENSOR  : rolling variance of a (de-trended) proxy for rho rises as the loyal
              well flattens (critical slowing down) -> trigger when it crosses
              theta x baseline.
  * ACTUATOR: on trigger, engage a nonsmooth deadband -> project rho onto
              K = [0, rho_cap] with rho_cap < rho_+ (removes the hacking basin).

Ramp stakes w past the open-loop fold w_+ and compare open vs closed loop.
CPU-only, numpy only.
"""
import numpy as np

rstar = 0.5
sigc = (np.sqrt(3) + 2*np.arctanh(1/np.sqrt(3))) / (8*rstar)


def A(rho, sig):
    return 0.5 + 0.5/(1 + np.exp(8*sig*(rho - rstar)))


def h(rho, sig):
    return sig/np.cosh(4*sig*(rho - rstar))**2


def dA(rho, sig, eps=1e-4):
    return (A(rho+eps, sig) - A(rho-eps, sig)) / (2*eps)


def wplus(sig, c=1.0):
    g = np.linspace(1e-4, rstar-1e-4, 200000)
    return (c*g/h(g, sig)).max()


def rho_plus(sig, c=1.0):
    g = np.linspace(1e-4, rstar-1e-4, 200000)
    return g[np.argmax(c*g/h(g, sig))]


def run_loop(sig, closed, c=1.0, mu=0.03, xi=0.012, seed=0, steps=16000,
             win=250, cap_frac=0.9, warmup=3000, var_trigger=3e-4):
    """Ramp w from 0.3 w_+ to 1.5 w_+. Returns (w/w_+, rho trajectory, engaged_step)."""
    wp = wplus(sig, c)
    rho_cap = cap_frac * rho_plus(sig, c)      # rho_cap < rho_+  (Theorem 2c)
    ws = np.linspace(0.3*wp, 1.5*wp, steps)
    rng = np.random.default_rng(seed)
    rho = 0.02
    engaged = None
    buf = []
    traj = np.empty(steps)
    for t in range(steps):
        w = ws[t]
        buf.append(rho)
        if len(buf) > win:
            buf.pop(0)
        if closed and engaged is None and len(buf) == win and t > warmup:
            seg = np.array(buf)
            seg = seg - np.polyval(np.polyfit(np.arange(win), seg, 1), np.arange(win))
            if seg.var() > var_trigger:
                engaged = t
        drift = mu * (w*h(rho, sig) - c*rho)   # dPi/drho = w h - c rho  (h = -A' > 0)
        rho = rho + drift + xi*np.sqrt(mu)*rng.standard_normal()
        hi = rho_cap if (closed and engaged is not None) else 1.0
        rho = min(hi, max(0.0, rho))           # project onto [0, hi]
        traj[t] = rho
    return ws/wp, traj, engaged


def main():
    print(f"sigma_c (single agent) = {sigc:.3f}")
    print("Ramping w past the open-loop fold w_+. open vs closed loop:\n")
    print(f"{'sigma':>6} {'open final':>11} {'closed final':>13} {'engaged':>9} {'success':>8}")
    for sig in (1.0, 1.4, 2.0):
        of = np.mean([run_loop(sig, False, seed=s)[1][-1] for s in range(6)])
        cfs, eng_ct = [], 0
        for s in range(6):
            _, tr, eng = run_loop(sig, True, seed=s)
            cfs.append(tr[-1]); eng_ct += (eng is not None)
        cf = np.mean(cfs); success = np.mean([f < 0.3 for f in cfs])
        print(f"{sig:>6.1f} {of:>11.3f} {cf:>13.3f} {eng_ct}/6{'':>3} {success:>8.2f}")
    print("\nExpected: open loop flips to hacking (~0.65-0.79);")
    print("          closed loop stays loyal (~0.08-0.12), engaged 6/6.")


if __name__ == "__main__":
    main()
