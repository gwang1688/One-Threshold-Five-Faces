import numpy as np
rstar=0.5; sigc=(np.sqrt(3)+2*np.arctanh(1/np.sqrt(3)))/(8*rstar)
def A(rho,sig): return 0.5+0.5/(1+np.exp(8*sig*(rho-rstar)))
def h(rho,sig): return sig/np.cosh(4*sig*(rho-rstar))**2
def wplus(sig,c=1.0): rho=np.linspace(1e-4,rstar-1e-4,200000); return (c*rho/h(rho,sig)).max()

# stochastic dynamics near the loyal well; slowly ramp w toward w_+; measure CSD (var, lag1-AC) in windows
def run_csd(sig, c=1.0, mu=0.04, noise=0.02, ramp_frac=0.9, seed=0, win=400):
    wp=wplus(sig,c); w_end=ramp_frac*wp        # ramp from low w up to just below the fold
    steps=20000; r=np.random.default_rng(seed); rho=0.02
    ws=np.linspace(0.2*wp, w_end, steps); traj=np.empty(steps)
    for t in range(steps):
        w=ws[t]; dA=-(A(rho+1e-4,sig)-A(rho-1e-4,sig))/2e-4
        rho=rho+mu*(w*dA-c*rho)+noise*np.sqrt(mu)*r.standard_normal()
        rho=np.clip(rho,0,1); traj[t]=rho
    # rolling CSD indicators on detrended residuals
    nwin=steps//win; var=np.zeros(nwin); ac=np.zeros(nwin); wmid=np.zeros(nwin)
    for i in range(nwin):
        seg=traj[i*win:(i+1)*win]; seg=seg-np.polyval(np.polyfit(np.arange(len(seg)),seg,1),np.arange(len(seg)))
        var[i]=seg.var()
        ac[i]=np.corrcoef(seg[:-1],seg[1:])[0,1] if seg.var()>1e-12 else 0
        wmid[i]=ws[i*win+win//2]
    return wmid,var,ac,wp

print(f"sigma_c={sigc:.3f}. CSD (var, lag1-AC) as w -> w_+ (fold). Does warning appear, and erode with sigma?")
print("theory: curvature d2Pi=-h*Psi' -> 0 at fold => var,AC rise. Thm4: sharper sigma => rho_+->0, w_+ huge => abrupt, less warning.\n")
print(f"{'sigma':>6} {'w_+':>9} {'rho_+':>7} {'var: far->near fold':>22} {'AC: far->near fold':>21} {'warns?':>7}")
for sig in [0.85,1.0,1.4,2.0,3.0]:
    rhog=np.linspace(1e-4,rstar-1e-4,200000); rp=rhog[np.argmax(rhog/h(rhog,sig))]
    wm,var,ac,wp=run_csd(sig)
    # CSD warning = var and AC rise in the last approach vs early. Compare first 20% vs last 20% of the ramp
    n=len(var); far_v=np.median(var[:n//5]); near_v=np.median(var[-n//5:])
    far_a=np.median(ac[:n//5]); near_a=np.median(ac[-n//5:])
    rise_v=near_v/max(far_v,1e-9); rise_a=near_a-far_a
    warns = "YES" if (rise_v>2.0 and near_a>far_a) else "weak" if rise_v>1.3 else "NO"
    print(f"{sig:>6} {wp:>9.2f} {rp:>7.3f} {far_v:>9.2e}->{near_v:.2e} {far_a:>8.2f}->{near_a:.2f} {warns:>7}")

print("\n=== warning LEAD: how close to w_+ before var doubles? (fraction of ramp remaining = warning window) ===")
for sig in [0.85,1.0,1.4,2.0,3.0]:
    wm,var,ac,wp=run_csd(sig)
    base=np.median(var[:len(var)//5])
    idx=np.argmax(var>2*base) if (var>2*base).any() else len(var)-1
    frac_remaining=(wp-wm[idx])/wp if idx<len(wm) else 0
    print(f" sigma={sig}: var doubles at w={wm[idx]:.2f} (w_+={wp:.2f}); warning window = {frac_remaining*100:.1f}% of w_+ remaining")
