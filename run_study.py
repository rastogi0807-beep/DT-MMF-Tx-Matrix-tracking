"""Run tuning, independent uncertainty calibration and paired-seed tests."""
import argparse, csv, json, os, sys, time, platform
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from twin_core import Fiber,Settings,run_filter,COLUMNS,truth_sequence,noise_vector,observation

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'
DATA.mkdir(exist_ok=True)


def save_csv(name,rows,columns=COLUMNS):
    with (DATA/name).open('w',newline='') as f:
        w=csv.writer(f);w.writerow(columns);w.writerows(rows)


def periodic(fiber,seed,cfg,scenario):
    s,ts,bad,onset=truth_sequence(fiber,seed,cfg,scenario)
    sigma=np.sqrt(np.mean(fiber.d**2)/(fiber.n*10**(cfg.snr/10)))
    interval=fiber.n//cfg.budget
    est=ts[0].copy(); rows=[]
    for t in range(cfg.frames):
        ref=np.exp(s[t,-2]+1j*s[t,-1])+noise_vector(seed,t,8000,1,.006)[0]
        if scenario=='anchor_fault' and onset<=t<onset+8:
            ref*=np.exp(.18+.60j)
        if t%interval==0:
            ys=np.column_stack([observation(fiber,s[t],ts[t],np.eye(fiber.n)[:,j],seed,t,9000+j,sigma,scenario,onset) for j in range(fiber.n)])
            est=ys/ref
        e=np.linalg.norm(est-ts[t])/np.linalg.norm(ts[t])
        pe=np.linalg.norm((est-ts[t])@fiber.eval_inputs)/np.linalg.norm(ts[t]@fiber.eval_inputs)
        rows.append([seed,t,e,pe,np.nan,np.nan,np.nan,np.nan,np.nan,0,1,int(bad[t]),np.nan,np.nan,0])
    return np.array(rows)


def numeric_checks():
    rng=np.random.default_rng(55)
    f=Fiber(654)
    z=rng.normal(0,.4,6)
    t,dt=f.matrix(z,True)
    h=np.einsum('k,kij->ij',z,f.gs)
    independent=f.ld@expm(1j*h)@f.right.conj().T
    eps=1e-6
    err=[]
    for k in range(6):
        zp=z.copy();zm=z.copy();zp[k]+=eps;zm[k]-=eps
        fd=(f.matrix(zp)-f.matrix(zm))/(2*eps)
        err.append(float(np.linalg.norm(fd-dt[k])/np.linalg.norm(fd)))
    sv=np.linalg.svd(t,compute_uv=False)
    s=np.r_[z,.04,.12]
    _,j=f.prediction(s,f.pilots[:,:3])
    fds=[]
    from twin_core import stack
    for k in range(8):
        sp=s.copy();sm=s.copy();sp[k]+=eps;sm[k]-=eps
        fds.append(stack((f.prediction(sp,f.pilots[:,:3],False)-f.prediction(sm,f.pilots[:,:3],False))/(2*eps)))
    joint_error=np.linalg.norm(j-np.array(fds).T)/np.linalg.norm(j)
    report={'derivative_relative_errors':err,'joint_jacobian_relative_error':float(joint_error),'independent_exponential_max_abs':float(np.max(np.abs(t-independent))),'singular_value_error':float(np.max(np.abs(sv-np.sort(f.d)[::-1]))),'largest_singular_value':float(sv[0])}
    assert max(err)<1e-6 and joint_error<1e-6
    assert report['independent_exponential_max_abs']<1e-12
    assert sv[0]<=1+1e-12
    (DATA/'numerical_checks.json').write_text(json.dumps(report,indent=2))
    return report


def calibrate():
    # Separate tuning seeds establish health gates; calibration seeds establish interval scale.
    cfg=Settings()
    healthy=[];nis=[]; raw=[]
    for seed in range(100,106):
        f=Fiber(10000+seed)
        r,a=run_filter(f,seed,cfg,'nominal','active',True)
        healthy.extend(r[12:,7]);nis.extend(r[12:,8]);raw.extend(r.tolist())
    cfg.gate=float(np.quantile(healthy,.995,method='higher'))
    cfg.alarm=float(np.quantile(nis,.995,method='higher'))
    # Multiplicative scaling from independent nominal trajectories, not test truth.
    ratios=[];raw_cal=[]
    for seed in range(200,210):
        f=Fiber(10000+seed)
        r,a=run_filter(f,seed,cfg,'nominal','active',True)
        ratios.extend(np.concatenate([x['standardized'] for x in a[12:]]))
        raw_cal.extend(r.tolist())
    cfg.kappa=max(1.,float(np.quantile(ratios,.95))/1.96)
    save_csv('tuning_trajectories.csv',raw)
    save_csv('calibration_trajectories.csv',raw_cal)
    out=vars(cfg)
    out.update({'tuning_seeds':list(range(100,106)),'calibration_seeds':list(range(200,210)),'test_seeds':list(range(1000,1024)),'interval_target':.95,'gate_quantile':.995,'calibration_status':'empirical nominal calibration; no distribution free guarantee under drift or adaptive acquisition'})
    (DATA/'configuration.json').write_text(json.dumps(out,indent=2))
    return cfg


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['pilot','calibrate','test','sensitivity','checks'],default='pilot');ap.add_argument('--seeds',type=int,default=24);args=ap.parse_args()
    start=time.perf_counter()
    if args.stage=='checks':
        print(json.dumps(numeric_checks(),indent=2));return
    if args.stage=='calibrate':
        print(json.dumps(vars(calibrate()),indent=2));return
    cfg=Settings()
    if (DATA/'configuration.json').exists():
        config=json.loads((DATA/'configuration.json').read_text())
        cfg=Settings(**{k:v for k,v in config.items() if k in vars(cfg)})
    scenarios=['nominal','jump','reference_jump','mismatch','corruption','anchor_fault','loss_mismatch']
    methods=['frozen','periodic','random','active','full','no_anchor','no_inflation']
    seeds=range(1000,1000+args.seeds)
    if args.stage=='pilot':
        seeds=[42];methods=['random','active','full','no_anchor','no_inflation','periodic'];scenarios=['nominal','jump','mismatch','corruption','anchor_fault']
    if args.stage in ('pilot','test'):
        summaries=[]; all_rows=[]
        for seed in seeds:
            f=Fiber(10000+seed)
            for scenario in scenarios:
                active_cache=None
                for method in methods:
                    if args.stage=='test' and method in ('no_anchor','no_inflation') and scenario not in ('nominal','jump','reference_jump'):
                        continue
                    st=time.perf_counter()
                    if method=='periodic':
                        rows=periodic(f,seed,cfg,scenario); arr=[]
                    elif method=='full' and active_cache is not None:
                        # The gate only controls publication of the estimate; fit and selection are identical.
                        rows=active_cache.copy();arr=[]
                        rows[:,10]=((rows[:,7]<=cfg.gate)&(rows[:,6]<=cfg.uncertainty_gate)).astype(int)
                    else:
                        retain=(args.stage=='pilot' or (seed==1000 and method=='active'))
                        rows,arr=run_filter(f,seed,cfg,scenario,method,retain)
                    if method=='active':active_cache=rows.copy()
                    for row in rows:
                        all_rows.append([scenario,method]+row.tolist())
                    good=rows[:,1]>=12
                    summaries.append([seed,scenario,method,float(np.mean(rows[good,2])),float(np.mean(rows[good,3])),float(np.mean(rows[good,10])),time.perf_counter()-st])
                    if arr and args.stage=='test':
                        fn=DATA/f'raw_{seed}_{scenario}_{method}.npz'
                        np.savez_compressed(fn,state=np.array([x['state'] for x in arr]),cov=np.array([x['cov'] for x in arr]),true_state=np.array([x['true_state'] for x in arr]),ids=np.array([x['ids'] for x in arr]),validation=np.array([x['validation'] for x in arr]),reference=np.array([x['reference'] for x in arr]),tm=np.array([x['tm'] for x in arr]),true_tm=np.array([x['true_tm'] for x in arr]),standardized=np.array([x['standardized'] for x in arr]))
                    print(seed,scenario,method,'TM',round(float(np.mean(rows[good,2])),5),'accepted',round(float(np.mean(rows[good,10])),3),'health',round(float(np.nanmean(rows[good,7])),2) if method!='periodic' else '-',flush=True)
            save_csv(args.stage+'_trajectories.csv',all_rows,['scenario','method']+COLUMNS)
            save_csv(args.stage+'_seed_summary.csv',summaries,['seed','scenario','method','tm_nrmse','field_nrmse','accepted_fraction','runtime_s'])
    if args.stage=='sensitivity':
        rows=[]
        for seed in range(3000,3012):
            for snr,budget in [(20,4),(25,4),(35,4),(40,4),(30,2),(30,8)]:
                c=Settings(**{**vars(cfg),'snr':snr,'budget':budget})
                for method in ['random','active','full']:
                    r,_=run_filter(Fiber(10000+seed),seed,c,'nominal',method)
                    for row in r:
                        rows.append([snr,budget,method]+row.tolist())
                print(seed,snr,budget,flush=True)
            save_csv('sensitivity_trajectories.csv',rows,['snr_db','budget','method']+COLUMNS)
    (DATA/f'{args.stage}_run_log.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'elapsed_s':time.perf_counter()-start,'settings':vars(cfg),'stage':args.stage},indent=2))
    print('Elapsed',time.perf_counter()-start,flush=True)


if __name__=='__main__':main()
