import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# Two-time-scale training toy:
#  slow var rho(t) in [0,1] = adversary pressure moving hack footprint into null channel (loyal->hacking)
#  fast var = observer estimating top eigenvector from n samples of the resolved covariance
#  s^2 = blatancy, gamma = m/n. Resolved spike beta = s^2 (1 - rho). BBP edge sqrt(gamma).
#  Duality gap: primal (adversary hides, value delivered) vs dual (observer's best detection).
#  Shadow price: marginal detection info per unit resolved energy = d(AUC)/d(beta) proxy; diverges as beta->edge.

def auc_at(beta, m, n, seeds=24, seed0=0):
    edge=(1+np.sqrt(m/n))**2; det=[]; cln=[]
    for s in range(seeds):
        r=np.random.default_rng(seed0+s+int(1000*beta)+m)
        var=np.ones(m); var[0]=1+max(beta,0)                 # planted resolved spike on 1 direction
        X=r.standard_normal((n,m))*np.sqrt(var); det.append(np.linalg.eigvalsh(X.T@X/n)[-1])
        Xc=r.standard_normal((n,m)); cln.append(np.linalg.eigvalsh(Xc.T@Xc/n)[-1])
    d=np.array(det); c=np.array(cln)
    x=np.concatenate([d,c]);o=np.argsort(x);rk=np.empty(len(x));rk[o]=np.arange(1,len(x)+1)
    return (rk[:len(d)].sum()-len(d)*(len(d)+1)/2)/(len(d)*len(c))

def overlap(beta,gamma):   # BBP recovery overlap (=0 below edge) -> primal recoverability
    return max((1-gamma/beta**2)/(1+gamma/beta),0) if beta>np.sqrt(gamma) else 0.0

fig,ax=plt.subplots(1,3,figsize=(15,4.4)); plt.subplots_adjust(wspace=0.28)
s2=1.6

# ---- (1) duality gap along the slow trajectory rho(t): sharp transition or smooth slope? sharpen with dim? ----
rhos=np.linspace(0,1,41)
for (m,n,mk,col) in [(60,120,'o','#4477aa'),(180,360,'s','#66aa55'),(540,1080,'^','#cc4433')]:
    gamma=m/n; gap=[]
    for rho in rhos:
        beta=s2*(1-rho)
        # primal value the adversary delivers undetected ~ (1 - detectability); dual = observer detection
        A=auc_at(beta,m,n); ov=overlap(beta,gamma)
        # duality gap proxy: |observer's achievable detection advantage - adversary's escaped value|
        dual=2*(A-0.5)            # observer detection advantage in [0,1]
        primal=1-ov               # adversary's un-recovered (escaped) fraction
        gap.append(abs(primal-dual))
    ax[0].plot(rhos,gap,mk+'-',ms=3,color=col,label=f"m={m} (γ={gamma:.2f})")
rho_star=1-np.sqrt((60/120))/s2
ax[0].axvline(rho_star,color='k',ls=':',lw=1,label=f"ρ* (BBP)")
ax[0].set_title("(1) Duality gap along ρ(t): sharp or smooth? sharpen with dim?",fontsize=10,fontweight='bold')
ax[0].set_xlabel("slow var ρ (adversary pressure, loyal→hacking)"); ax[0].set_ylabel("|primal − dual| gap"); ax[0].legend(fontsize=8)

# ---- (2) shadow price = d(detection)/d(resolved energy): does it diverge at the wall? ----
for (m,n,col) in [(60,120,'#4477aa'),(180,360,'#66aa55'),(540,1080,'#cc4433')]:
    gamma=m/n; betas=np.linspace(0.02,s2,60); A=[auc_at(b,m,n) for b in betas]
    A=np.array(A); sp=np.gradient(A,betas)          # marginal detection per unit resolved energy
    ax[1].plot(betas,sp,'-',color=col,label=f"m={m}")
ax[1].axvline(np.sqrt(60/120),color='k',ls=':',lw=1,label="edge √γ")
ax[1].set_title("(2) Shadow price d(AUC)/d(resolved energy): diverges at wall?",fontsize=10,fontweight='bold')
ax[1].set_xlabel("resolved spike β = s²(1−ρ)"); ax[1].set_ylabel("shadow price (marginal detection)"); ax[1].legend(fontsize=8)

# ---- (3) two-time-scale run: does shadow price LEAD the AUC collapse (early warning)? ----
T=200; rho_traj=np.clip(np.linspace(-0.2,1.1,T),0,1)   # adversary slowly drifts loyal->hacking
m,n=180,360; gamma=m/n
auc_t=[]; sp_t=[]; beta_prev=None; A_prev=None
for t,rho in enumerate(rho_traj):
    beta=s2*(1-rho); A=auc_at(beta,m,n,seeds=16)
    auc_t.append(A)
    if A_prev is not None and beta_prev is not None and beta!=beta_prev:
        sp_t.append(abs((A-A_prev)/(beta-beta_prev)))
    else: sp_t.append(0)
    A_prev=A; beta_prev=beta
auc_t=np.array(auc_t); sp_t=np.array(sp_t)
axL=ax[2]; axR=axL.twinx()
axL.plot(np.arange(T),auc_t,'-',color='#cc4433',lw=2,label="detection AUC")
axR.plot(np.arange(T),sp_t,'-',color='#4477aa',lw=1.5,label="shadow price")
# mark AUC collapse (crosses 0.6) and shadow-price peak
collapse=np.argmax(auc_t<0.6) if (auc_t<0.6).any() else T-1
sp_peak=np.argmax(sp_t)
axL.axvline(collapse,color='#cc4433',ls='--',lw=1); axR.axvline(sp_peak,color='#4477aa',ls=':',lw=1.5)
axL.set_title(f"(3) Two-time-scale: shadow-price peak (t={sp_peak}) vs AUC collapse (t={collapse})",fontsize=10,fontweight='bold')
axL.set_xlabel("training step t (ρ drifts loyal→hacking)"); axL.set_ylabel("AUC",color='#cc4433'); axR.set_ylabel("shadow price",color='#4477aa')
axL.legend(loc='lower left',fontsize=8); axR.legend(loc='upper right',fontsize=8)

plt.savefig("intent_drift.png",dpi=120,bbox_inches='tight')
print("saved. diagnostics:")
print(f" (3) shadow-price peak at t={sp_peak}, AUC-collapse at t={collapse}, lead={collapse-sp_peak} steps (>0 => early warning)")
# sharpness of gap transition across dims
for m,n in [(60,120),(180,360),(540,1080)]:
    gamma=m/n; b1=s2*(1-(rho_star-0.08)); b2=s2*(1-(rho_star+0.08))
    print(f"   m={m}: gap slope across ρ*±0.08 ~ {abs((1-overlap(b2,gamma))-(1-overlap(b1,gamma))):.3f}")
