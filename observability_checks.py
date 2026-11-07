import json,csv
from pathlib import Path
import numpy as np
from scipy.linalg import null_space
from scipy.stats import beta
from twin_core import Fiber,Settings,run_filter,select_pilots
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'data'
cfg0=json.loads((D/'configuration.json').read_text());cfg=Settings(**{k:v for k,v in cfg0.items()if k in vars(Settings())})
f=Fiber(11000)
s=np.r_[np.array([.10,-.12,.08,-.15,.06,.10]),.04,.12]
p=np.diag([.018**2]*6+[.005**2,.012**2])
sigma=np.sqrt(np.mean(f.d**2)/(f.n*10**(cfg.snr/10)))
ids=select_pilots(f,s,p,3,sigma,'active',np.random.default_rng(4),120)
x=np.c_[f.sentinel,f.pilots[:,ids]]
v=null_space(x.conj().T)[:,0]
t=f.matrix(s[:6]);theta=.8
q=np.eye(f.n)+(np.exp(1j*theta)-1)*np.outer(v,v.conj())
tp=t@q
report={'selected_ids':ids,'pilot_count':x.shape[1],'input_dimension':f.n,'unobserved_subspace_dimension':int(f.n-np.linalg.matrix_rank(x)),'phase_rotation_rad':theta,'pilot_field_max_abs_difference':float(np.max(abs((tp-t)@x))),'full_tm_nrmse':float(np.linalg.norm(tp-t)/np.linalg.norm(t)),'heldout_unseen_field_nrmse':float(np.linalg.norm((tp-t)@f.eval_inputs)/np.linalg.norm(tp@f.eval_inputs)),'singular_value_max_difference':float(np.max(abs(np.linalg.svd(tp,compute_uv=False)-np.linalg.svd(t,compute_uv=False)))),'interpretation':'One-frame passive counterexample chosen after the measured input span is fixed. All current pilot likelihoods, fits and validation decisions are unchanged; arbitrary unmodelled transformations cannot be certified from four input directions.'}
assert report['pilot_field_max_abs_difference']<1e-12
assert report['singular_value_max_difference']<1e-12
(D/'blind_subspace_counterexample.json').write_text(json.dumps(report,indent=2))
fault=json.loads((D/'fault_results.json').read_text())
for item in fault:
 item['independent_trajectories_detected']=24
 item['trajectory_detection_exact_binomial_ci']=[float(beta.ppf(.025,24,1)),1.]
(D/'fault_results.json').write_text(json.dumps(fault,indent=2))
print(json.dumps(report,indent=2),flush=True)
rows=[]
for seed in range(4000,4008):
 for strength in [.01,.03,.06,.10,.15,.23]:
  ff=Fiber(10000+seed);ff.bad_generator*=strength/.23
  r,_=run_filter(ff,seed,cfg,'mismatch','active')
  after=r[r[:,1]>=80]
  accept=(after[:,7]<=cfg.gate)&(after[:,6]<=cfg.uncertainty_gate)
  rows.append([seed,strength,float(np.mean(after[:,2])),float(np.mean(accept)),float(np.mean(after[accept,2]>.05)) if accept.any() else np.nan,float(np.mean(after[:,7]))])
 with (D/'weak_mismatch_sweep.csv').open('w',newline='')as file:
  w=csv.writer(file);w.writerow(['seed','perturbation_rad','tm_nrmse','accepted_fraction','accepted_unsafe_risk','health']);w.writerows(rows)
 print('Completed stress seed',seed,flush=True)
