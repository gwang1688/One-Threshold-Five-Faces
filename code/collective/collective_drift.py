import numpy as np
rstar=0.5; sigc=(np.sqrt(3)+2*np.arctanh(1/np.sqrt(3)))/(8*rstar)
def A(rho,sig): return 0.5+0.5/(1+np.exp(8*sig*(rho-rstar)))
def h(rho,sig): return sig/np.cosh(4*sig*(rho-rstar))**2

# N coupled agents: effective stakes of agent i rises with mean hacking (herd / monitor-dilution)
#  d rho_i = mu ( w_eff * h(rho_i) - c rho_i ),  w_eff = w + kappa * mean(rho)
def run(N, kappa, sig, w0=0.6, c=1.0, mu=0.05, steps=6000, seed=0, rho0=None):
    r=np.random.default_rng(seed)
    rho = r.uniform(0,0.05,N) if rho0 is None else rho0.copy()
    for t in range(steps):
        mbar=rho.mean(); weff=w0+kappa*mbar
        dA=-(A(rho+1e-4,sig)-A(rho-1e-4,sig))/2e-4
        rho=np.clip(rho+mu*(weff*dA - c*rho),0,1)
    return rho

print(f"sigma_c={sigc:.3f}. Per-agent detector sigma=1.2 (catastrophic). w0=0.6 (below single-agent fold).")
print("Coupling kappa (herd/monitor-dilution). Does the collective avalanche to all-hacking at a critical kappa*?")
sig=1.2; N=200
print(f"\n{'kappa':>7} {'mean rho':>9} {'frac hacking':>13} {'regime':>22}")
for kap in [0.0,0.5,1.0,1.5,2.0,2.5,3.0,4.0]:
    rho=run(N,kap,sig)
    mbar=rho.mean(); frac=(rho>0.5).mean()
    reg = "all loyal" if frac<0.05 else ("all hacking (avalanche)" if frac>0.95 else "mixed")
    print(f"{kap:>7} {mbar:>9.3f} {frac:>13.2f} {reg:>22}")

print("\n=== (a) is the collective transition SHARP, and (b) does it sharpen with N? ===")
print("scan kappa finely near the jump; measure transition width in kappa, for several N")
sig=1.2
for N in [50,200,800]:
    kaps=np.linspace(1.0,3.5,60); fr=[]
    for kap in kaps:
        fr.append((run(N,kap,sig,seed=1)>0.5).mean())
    fr=np.array(fr)
    lo=kaps[np.argmax(fr>0.1)] if (fr>0.1).any() else np.nan
    hi=kaps[np.argmax(fr>0.9)] if (fr>0.9).any() else np.nan
    print(f" N={N:>4}: transition kappa in [{lo:.3f},{hi:.3f}], width={hi-lo:.3f}")

print("\n=== (c) HYSTERESIS: is the collective flip irreversible (avalanche up-kappa vs down-kappa)? ===")
sig=1.2; N=200
kaps=np.linspace(0.5,4.0,40)
up=[];rho=np.zeros(N)+0.02
for kap in kaps: rho=run(N,kap,sig,rho0=rho,steps=1500); up.append(rho.mean())
dn=[];rho=np.zeros(N)+0.9
for kap in kaps[::-1]: rho=run(N,kap,sig,rho0=rho,steps=1500); dn.append(rho.mean())
dn=dn[::-1]
print(f" up-sweep mean-rho jump at kappa~{kaps[np.argmax(np.diff(up))]:.2f}, down-sweep at kappa~{kaps[np.argmax(np.diff(dn))]:.2f}")
print(f" collective hysteresis width (max |up-down|) = {np.max(np.abs(np.array(up)-np.array(dn))):.3f}")
