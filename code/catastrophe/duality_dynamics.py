import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
s2=1.7
def A_rho(rho,m,n,seeds=30,seed0=0):   # detection AUC at pressure rho (BBP, sharpens with dim)
    beta=s2*max(1-rho,0.0); edge=(1+np.sqrt(m/n))**2; d=[];c=[]
    for s in range(seeds):
        r=np.random.default_rng(seed0+s+int(1000*rho)+m)
        var=np.ones(m); var[0]=1+beta
        X=r.standard_normal((n,m))*np.sqrt(var); d.append(np.linalg.eigvalsh(X.T@X/n)[-1])
        Xc=r.standard_normal((n,m)); c.append(np.linalg.eigvalsh(Xc.T@Xc/n)[-1])
    d=np.array(d);c=np.array(c);x=np.concatenate([d,c]);o=np.argsort(x);rk=np.empty(len(x));rk[o]=np.arange(1,len(x)+1)
    return (rk[:len(d)].sum()-len(d)*(len(d)+1)/2)/(len(d)*len(c))

def build_A(m,n,grid):
    return np.array([A_rho(r,m,n) for r in grid])

def eq_rho(Agrid, grid, w, c, rho_init):
    # gradient ASCENT to nearest local max of Pi(rho)=w(1-A)-0.5 c rho^2, starting from rho_init
    dA=np.gradient(Agrid,grid)
    rho=rho_init
    for _ in range(400):
        i=np.clip(np.searchsorted(grid,rho),1,len(grid)-1)
        dAi=dA[i]; grad = -w*dAi - c*rho     # dPi/drho
        rho=np.clip(rho+0.02*grad,0,1)
    return rho

fig,ax=plt.subplots(1,3,figsize=(15,4.4)); plt.subplots_adjust(wspace=0.3)
grid=np.linspace(0,1,60); c=1.2
dims=[(40,80,'#4477aa'),(160,320,'#66aa55'),(640,1280,'#cc4433')]
ws=np.linspace(0.2,3.0,40)

# ---- (1) the payoff landscape: unimodal (gradual) or bistable (abrupt) as dim grows? ----
for (m,n,col) in dims:
    Ag=build_A(m,n,grid)
    w_show=1.6
    Pi=w_show*(1-Ag)-0.5*c*grid**2
    ax[0].plot(grid,Pi-Pi.max(),color=col,lw=2,label=f"m={m}")
ax[0].set_title("(1) Adversary payoff Π(ρ) at fixed w: one hump or two?",fontsize=10,fontweight='bold')
ax[0].set_xlabel("intent ρ (loyal→hacking)"); ax[0].set_ylabel("Π (shifted)"); ax[0].legend(fontsize=8)

# ---- (2) hysteresis: sweep w up then down, track eq ρ. Loop = abrupt catastrophe ----
hyst_width={}
for (m,n,col) in dims:
    Ag=build_A(m,n,grid)
    up=[]; rho=0.0
    for w in ws: rho=eq_rho(Ag,grid,w,c,rho); up.append(rho)
    down=[]; rho=1.0
    for w in ws[::-1]: rho=eq_rho(Ag,grid,w,c,rho); down.append(rho)
    down=down[::-1]
    ax[1].plot(ws,up,'-',color=col,lw=2,label=f"m={m} up")
    ax[1].plot(ws,down,'--',color=col,lw=1.5)
    hyst_width[m]=np.max(np.abs(np.array(up)-np.array(down)))
ax[1].set_title("(2) Hysteresis in ρ vs stakes w (solid=up, dash=down)",fontsize=10,fontweight='bold')
ax[1].set_xlabel("stakes w (grows over training)"); ax[1].set_ylabel("equilibrium intent ρ*"); ax[1].legend(fontsize=7)

# ---- (3) jump size vs dimension: does the abrupt flip emerge/sharpen with d? ----
ms=[40,80,160,320,640]; jumps=[]; widths=[]
for m in ms:
    n=2*m; Ag=build_A(m,n,grid)
    up=[]; rho=0.0
    for w in ws: rho=eq_rho(Ag,grid,w,c,rho); up.append(rho)
    up=np.array(up); jumps.append(np.max(np.diff(up)))       # largest single-step jump in ρ
    down=[]; rho=1.0
    for w in ws[::-1]: rho=eq_rho(Ag,grid,w,c,rho); down.append(rho)
    widths.append(np.max(np.abs(up-np.array(down[::-1]))))
ax[2].plot(ms,jumps,'o-',color='#cc4433',lw=2,label="max jump in ρ (per Δw)")
ax[2].plot(ms,widths,'s-',color='#4477aa',lw=2,label="hysteresis width")
ax[2].axhline(0.05,color='gray',ls=':',lw=1)
ax[2].set_title("(3) Does the abrupt flip emerge with dimension?",fontsize=10,fontweight='bold')
ax[2].set_xlabel("dimension m"); ax[2].set_ylabel("jump / hysteresis in ρ"); ax[2].set_xscale('log'); ax[2].legend(fontsize=8)
plt.savefig("duality_dynamics.png",dpi=120,bbox_inches='tight')

print("=== VERDICT DIAGNOSTICS ===")
for m in ms: pass
print("hysteresis width vs dim:", {m:round(w,3) for m,w in zip(ms,widths)})
print("max jump vs dim:        ", {m:round(j,3) for m,j in zip(ms,jumps)})
print("(hysteresis/jump GROWING with m => cliff induces catastrophe = NEW; flat/zero => gradual slide)")
