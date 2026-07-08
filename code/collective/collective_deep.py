import numpy as np
rstar=0.5; sigc=(np.sqrt(3)+2*np.arctanh(1/np.sqrt(3)))/(8*rstar)
def A(rho,sig): return 0.5+0.5/(1+np.exp(8*sig*(rho-rstar)))
def h(rho,sig): return sig/np.cosh(4*sig*(rho-rstar))**2
def rho_eq(W, sig, c, start=0.0):        # single-agent stable equilibrium at stakes W, cost c, continued from start
    rho=start
    for _ in range(1500):
        dA=-(A(rho+1e-4,sig)-A(rho-1e-4,sig))/2e-4
        rho=np.clip(rho+0.05*(W*dA-c*rho),0,1)
    return rho
def single_fold_wplus(sig,c):            # single-agent up-threshold w_+ for cost c (loyal fold)
    rho=np.linspace(1e-4,rstar-1e-4,20000); return (c*rho/h(rho,sig)).max()

sig=1.2; w0=0.6
# ---- HETEROGENEOUS MEAN-FIELD self-consistency ----
# m = E_c[ rho_eq(w0 + kappa*m ; c) ] ; loyal branch loses stability -> collective kappa*
# Claim: kappa* is a functional of the cost distribution, DOMINATED BY THE LOWER TAIL (weakest agents).
def kappa_star(cost_samples, sig=1.2, w0=0.6):
    cs=np.asarray(cost_samples)
    kaps=np.linspace(0,8,200); m=0.02
    for kap in kaps:
        for _ in range(60):
            m=np.mean([rho_eq(w0+kap*m,sig,c,start=min(rho_eq(w0+kap*m,sig,c),0.05)) for c in cs[:60]])
        if m>0.3: return kap
    return np.nan

print(f"sigma={sig}, w0={w0}, single-agent sigma_c={sigc:.3f}")
print("=== (1) is kappa* dominated by the LOWER TAIL of the cost distribution? ===")
print("Fix the MEAN cost=1.0; vary only the LOWER TAIL (min cost), keep mean fixed. If kappa* moves -> tail-dominated.")
rng=np.random.default_rng(0)
print(f"{'min cost (tail)':>16} {'mean cost':>10} {'kappa*':>8}")
for cmin in [1.0,0.7,0.5,0.35,0.25]:
    # construct distribution with fixed mean 1.0 but varying lower tail cmin
    cs=np.concatenate([np.full(20,cmin), np.full(80,(1.0*100-cmin*20)/80)])  # mean stays 1.0
    ks=kappa_star(cs)
    print(f"{cmin:>16.2f} {cs.mean():>10.2f} {ks:>8.2f}")
print(" -> if kappa* falls sharply as the tail (cmin) drops at FIXED mean: WEAKEST-LINK law confirmed.")

print("\n=== predicted weakest-link scaling: kappa* set by when the WEAKEST agents' fold is crossed ===")
print("weakest agent (cost cmin) self-ignites when w0+kappa*m_seed > w_+(cmin); compare kappa* to w_+(cmin) form")
for cmin in [1.0,0.7,0.5,0.35]:
    wp=single_fold_wplus(sig,cmin)
    print(f"  cmin={cmin}: single-agent w_+({cmin})={wp:.3f}  (lower cost -> lower fold -> seeds earlier)")
