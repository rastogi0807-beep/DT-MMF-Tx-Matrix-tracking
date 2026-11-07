"""Reproducible reduced coupled-mode MMF study; all observations are synthetic."""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh, expm


def stack(a):
    a = np.asarray(a).ravel(order='F')
    return np.r_[a.real, a.imag]


def unstack(a, shape):
    n = len(a)//2
    return (a[:n]+1j*a[n:]).reshape(shape, order='F')


def unitary(rng,n):
    q,r=np.linalg.qr(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))
    return q * (np.diag(r)/np.abs(np.diag(r))).conj()[None,:]


@dataclass
class Settings:
    n: int = 32
    p: int = 6
    frames: int = 160
    budget: int = 4          # Includes one independent validation pilot.
    snr: float = 30.0
    qz: float = .018
    qa: float = .005
    qphase: float = .012
    initial_std: float = .035
    iterations: int = 3
    kappa: float = 1.0
    gate: float = 10.0
    uncertainty_gate: float = .12
    alarm: float = 2.0
    inflation: float = .45


class Fiber:
    def __init__(self,seed,n=32,p=6):
        rng=np.random.default_rng(seed)
        self.n,self.p=n,p
        self.left=unitary(rng,n)
        self.right=unitary(rng,n)
        self.d=np.linspace(.75,.98,n)
        self.ld=self.left*self.d[None,:]
        gs=[]
        for k in range(p):
            g=np.zeros((n,n),complex)
            start=int(k*(n-4)/(p-1))
            ix=np.arange(start,start+4)
            local=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4))
            local=(local+local.conj().T)/2
            local-=np.trace(local)*np.eye(4)/4
            g[np.ix_(ix,ix)]=local
            g*=np.sqrt(2)/np.linalg.norm(g)
            gs.append(g)
        self.gs=np.array(gs)
        # Broad, independent validation input; never included in the MAP fit.
        sent=np.exp(2j*np.pi*rng.random(n))/np.sqrt(n)
        self.sentinel=self.right@sent
        # n mode-selective and n broad phase-coded feasible complex inputs.
        broad=np.exp(2j*np.pi*rng.random((n,n)))/np.sqrt(n)
        self.pilots=self.right@np.c_[np.eye(n),broad]
        self.eval_inputs=self.right@(np.exp(2j*np.pi*rng.random((n,8)))/np.sqrt(n))
        self.reference=1.+0j
        self.noise=np.sqrt(np.mean(self.d**2)/(n*10**(30/10)))
        # A unitary perturbation outside the learned six-generator tangent.
        a=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
        a=(a+a.conj().T)/2
        a-=np.trace(a)*np.eye(n)/n
        for g in self.gs:
            a-=np.vdot(g,a).real/np.vdot(g,g).real*g
        self.bad_generator=a/np.linalg.norm(a)*np.sqrt(n)

    def matrix(self,z,derivatives=False):
        h=np.einsum('k,kij->ij',z,self.gs)
        lam,v=eigh(h,check_finite=False)
        e=np.exp(1j*lam)
        u=(v*e[None,:])@v.conj().T
        t=self.ld@u@self.right.conj().T
        if not derivatives:
            return t
        # Stable divided difference including repeated eigenvalues.
        delta=lam[:,None]-lam[None,:]
        mid=(lam[:,None]+lam[None,:])/2
        f=1j*np.exp(1j*mid)*np.sinc(delta/(2*np.pi))
        lv=self.ld@v
        vr=v.conj().T@self.right.conj().T
        dg=np.array([lv@(f*(v.conj().T@g@v))@vr for g in self.gs])
        return t,dg

    def prediction(self,s,x,derivatives=True):
        if derivatives:
            t,dt=self.matrix(s[:self.p],True)
        else:
            t=self.matrix(s[:self.p])
        gain=np.exp(s[-2]+1j*s[-1])
        y=gain*t@x
        if not derivatives:
            return y
        jz=np.array([stack(gain*d@x) for d in dt]).T
        j=np.c_[jz,stack(y),stack(1j*y)]
        return y,j


def truth_sequence(fiber,seed,cfg,scenario='nominal'):
    key=(seed,cfg.frames,scenario)
    if not hasattr(fiber,'_truth_cache'):
        fiber._truth_cache={}
    if key in fiber._truth_cache:
        return fiber._truth_cache[key]
    rng=np.random.default_rng(seed)
    n,p,t=cfg.n,cfg.p,cfg.frames
    z=np.zeros((t,p))
    steps=np.array([.010,.016,.008,.022,.012,.018])[:p]
    z[0]=rng.normal(0,.04,p)
    for k in range(1,t):
        z[k]=z[k-1]+rng.normal(size=p)*steps
    tt=np.arange(t)
    z+=.08*np.sin(tt[:,None]/(15+np.arange(p)*4)+np.arange(p))
    a=.04*np.sin(tt/31)+np.cumsum(rng.normal(0,.002,t))
    ph=.12*np.sin(tt/24)+np.cumsum(rng.normal(0,.006,t))
    onset=t//2
    if scenario in ('jump','combined'):
        z[onset:]+=np.array([.38,-.30,.40,-.26,.32,-.34])[:p]
    if scenario=='reference_jump':
        a[onset:]+=.15
        ph[onset:]+=.45
    s=np.c_[z,a,ph]
    ts=[]
    bad=np.zeros(t,bool)
    # Truth is generated independently of the estimator's exponential evaluator.
    for k in range(t):
        h=np.einsum('k,kij->ij',z[k],fiber.gs)
        u=expm(1j*h)
        if scenario in ('mismatch','combined') and k>=onset:
            u=expm(.23j*fiber.bad_generator)@u
            bad[k]=True
        d=fiber.d.copy()
        if scenario=='loss_mismatch' and k>=onset:
            d[:n//3]*=.70
            bad[k]=True
        ts.append((fiber.left*d[None,:])@u@fiber.right.conj().T)
    if scenario=='corruption':
        bad[onset:onset+8]=True
    if scenario=='anchor_fault':
        bad[onset:onset+8]=True
    out=(s,np.array(ts),bad,onset)
    fiber._truth_cache[key]=out
    return out


def noise_vector(seed,frame,kind,n,sigma):
    # Method-independent potential observations; pairing remains exact when selected inputs differ.
    rng=np.random.default_rng(np.random.SeedSequence([int(seed),int(frame),int(kind)]))
    return sigma/np.sqrt(2)*(rng.normal(size=n)+1j*rng.normal(size=n))


def observation(fiber,true_s,true_t,x,seed,frame,kind,sigma,scenario,onset):
    y=np.exp(true_s[-2]+1j*true_s[-1])*true_t@x
    y=y+noise_vector(seed,frame,kind,fiber.n,sigma)
    if scenario=='corruption' and onset<=frame<onset+8:
        y+=noise_vector(seed,frame,kind+10000,fiber.n,.12)
    return y


def reference_update(s,p,c,sigma,enabled=True):
    if not enabled:
        return s,p
    gain=np.exp(s[-2]+1j*s[-1])
    h=np.zeros((2,len(s)))
    h[:,-2]=[gain.real,gain.imag]
    h[:,-1]=[-gain.imag,gain.real]
    r=stack(c-gain)
    # Information form avoids a large observation-space matrix inverse.
    prec=np.linalg.inv(p)
    pn=np.linalg.inv(prec+h.T@h/(sigma**2/2))
    sn=s+pn@h.T@r/(sigma**2/2)
    return sn,(pn+pn.T)/2


def select_pilots(fiber,s,p,budget,sigma,mode,rng,frame):
    m=fiber.pilots.shape[1]
    if mode=='random':
        return rng.choice(m,budget,replace=False).tolist()
    if mode=='cyclic':
        return [int((frame*budget+k)%m) for k in range(budget)]
    t,dt=fiber.matrix(s[:fiber.p],True)
    g=np.exp(s[-2]+1j*s[-1])
    # TM prediction risk is invariant to gain. W also accounts for all cross covariances.
    dmat=np.array([stack(d) for d in dt]).T
    w=np.zeros_like(p)
    w[:fiber.p,:fiber.p]=dmat.T@dmat/np.linalg.norm(t)**2
    info=[]
    for x in fiber.pilots.T:
        y=g*t@x
        j=np.c_[np.array([stack(g*d@x) for d in dt]).T,stack(y),stack(1j*y)]
        info.append(j.T@j/(sigma**2/2))
    info=np.array(info)
    precision=np.linalg.inv(p)
    chosen=[]
    for _ in range(budget):
        covs=np.linalg.inv(precision[None,:,:]+info)
        risk=np.einsum('ij,kji->k',w,covs)
        risk[chosen]=np.inf
        idx=int(np.argmin(risk))
        chosen.append(idx)
        precision+=info[idx]
    return chosen


def map_update(fiber,s0,p0,xs,ys,sigma,iterations=3):
    s=s0.copy()
    precision=np.linalg.inv(p0)
    nr=sigma**2/2
    yvec=stack(ys)
    def objective(st):
        rr=yvec-stack(fiber.prediction(st,xs,False))
        ds=st-s0
        return rr@rr/nr+ds@precision@ds
    last=objective(s)
    used=0
    for it in range(iterations):
        pred,j=fiber.prediction(s,xs,True)
        resid=yvec-stack(pred)
        h=precision+j.T@j/nr
        step=np.linalg.solve(h,j.T@resid/nr-precision@(s-s0))
        scale=1.
        for _ in range(7):
            proposal=s+scale*step
            val=objective(proposal)
            if np.isfinite(val) and val<=last+1e-9:
                s=proposal;last=val;break
            scale*=.5
        used+=1
        if np.linalg.norm(scale*step)<1e-7:
            break
    _,j=fiber.prediction(s,xs,True)
    pn=np.linalg.inv(precision+j.T@j/nr)
    return s,(pn+pn.T)/2,used


def run_filter(fiber,seed,cfg,scenario,method,retain=False):
    true_s,true_t,bad,onset=truth_sequence(fiber,seed,cfg,scenario)
    rng=np.random.default_rng(seed+92371)
    p=cfg.p
    sigma=np.sqrt(np.mean(fiber.d**2)/(fiber.n*10**(cfg.snr/10)))
    sigma_ref=.006
    q=np.diag([cfg.qz**2]*p+[cfg.qa**2,cfg.qphase**2])
    initial_noise=noise_vector(seed,0,999,p+2,1).real
    s=true_s[0]+cfg.initial_std*np.sqrt(2)*initial_noise
    cov=np.eye(p+2)*cfg.initial_std**2
    frozen=s.copy()
    rows=[];arrays=[]
    for frame in range(cfg.frames):
        if method=='frozen':
            s=frozen.copy()
            cov=np.eye(p+2)*cfg.initial_std**2
        else:
            cov+=q
        c=np.exp(true_s[frame,-2]+1j*true_s[frame,-1])+noise_vector(seed,frame,8000,1,sigma_ref)[0]
        if scenario=='anchor_fault' and onset<=frame<onset+8:
            c*=np.exp(.18+.60j)
        anchor=(method not in ('no_anchor','frozen'))
        s,cov=reference_update(s,cov,np.array([c]),sigma_ref,anchor)
        ys=observation(fiber,true_s[frame],true_t[frame],fiber.sentinel,seed,frame,7000,sigma,scenario,onset)
        pred_s,js=fiber.prediction(s,fiber.sentinel)
        r=stack(ys-pred_s)
        # Woodbury NIS in the low-dimensional state space.
        nr=sigma**2/2
        mid=np.linalg.inv(np.linalg.inv(cov)+js.T@js/nr)
        nis=float((r@r/nr-(js.T@r)@mid@(js.T@r)/nr**2)/(2*fiber.n))
        alarm=nis>cfg.alarm
        if alarm and method in ('full','active','random','cyclic','no_anchor'):
            cov[:p,:p]+=np.eye(p)*cfg.inflation**2
        prior_s=s.copy();prior_cov=cov.copy()
        ids=[]
        if method!='frozen':
            mode='random' if method=='random' else ('cyclic' if method=='cyclic' else 'active')
            ids=select_pilots(fiber,s,cov,cfg.budget-1,sigma,mode,rng,frame)
            xs=fiber.pilots[:,ids]
            yp=np.column_stack([observation(fiber,true_s[frame],true_t[frame],xs[:,j],seed,frame,ids[j]+1,sigma,scenario,onset) for j in range(len(ids))])
            s,cov,iterations=map_update(fiber,s,cov,xs,yp,sigma,cfg.iterations)
        else:
            iterations=0
        # The independent validation pilot is never fitted, and is checked after updating.
        valid_pred,jv=fiber.prediction(s,fiber.sentinel)
        rv=stack(ys-valid_pred)
        predvar=np.sum((jv@cov)*jv,axis=1)+nr
        health=float(np.mean(rv**2/predvar))
        te,dt=fiber.matrix(s[:p],True)
        dmat=np.array([stack(d) for d in dt]).T
        risk=float(np.einsum('ij,ji->',cov[:p,:p],dmat.T@dmat)/np.linalg.norm(te)**2)
        uncertainty=cfg.kappa*np.sqrt(max(risk,0))
        accept=(health<=cfg.gate and uncertainty<=cfg.uncertainty_gate) if method=='full' else True
        # Full state is retained for recovery. Rejection blocks output, not state propagation.
        tmerr=np.linalg.norm(te-true_t[frame])/np.linalg.norm(true_t[frame])
        pred=te@fiber.eval_inputs
        truth=true_t[frame]@fiber.eval_inputs
        pe=np.linalg.norm(pred-truth)/np.linalg.norm(truth)
        jac=np.array([stack(d@fiber.eval_inputs) for d in dt]).T
        var=np.maximum(np.sum((jac@cov[:p,:p])*jac,axis=1),1e-16)
        errors=stack(pred-truth)
        standardized=np.abs(errors)/np.sqrt(var)
        coverage=np.mean(standardized<=1.96*cfg.kappa)
        width=float(np.mean(3.92*cfg.kappa*np.sqrt(var)))
        rows.append([seed,frame,tmerr,pe,coverage,width,uncertainty,health,nis,int(alarm),int(accept),int(bad[frame]),float(np.exp(s[-2])-np.exp(true_s[frame,-2])),float(s[-1]-true_s[frame,-1]),iterations])
        if retain:
            arrays.append(dict(state=s.copy(),cov=cov.copy(),true_state=true_s[frame].copy(),ids=np.array(ids),validation=ys.copy(),reference=c,standardized=standardized,tm=te.copy(),true_tm=true_t[frame].copy()))
    return np.array(rows),arrays


COLUMNS=['seed','frame','tm_nrmse','field_nrmse','interval_coverage','interval_width','tm_uncertainty','health','nis','drift_alarm','accepted','bad_frame','gain_error','phase_error','map_iterations']
