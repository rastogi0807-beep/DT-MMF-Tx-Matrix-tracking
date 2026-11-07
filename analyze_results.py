import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'data'
r=pd.read_csv(D/'test_trajectories.csv')
assert r.seed.nunique()==24, 'Run all 24 planned test seeds before final aggregation.'
cfg=json.loads((D/'configuration.json').read_text())
r['unsafe']=r.tm_nrmse>.05
r['rejected']=1-r.accepted
post=r[r.frame>=12].copy()

def ci(vals,rng=np.random.default_rng(9473)):
 a=np.asarray(vals,float)
 if not len(a):return [None,None]
 means=np.mean(rng.choice(a,(2000,len(a)),replace=True),axis=1)
 return np.quantile(means,[.025,.975]).tolist()

summary=[]
for (scenario,method),g in post.groupby(['scenario','method'],sort=False):
 seeds=g.groupby('seed').agg(tm=('tm_nrmse','mean'),field=('field_nrmse','mean'),accepted=('accepted','mean'),interval=('interval_coverage','mean'))
 ac=g[g.accepted==1]
 su={'scenario':scenario,'method':method,'seeds':len(seeds),'tm_nrmse':seeds.tm.mean(),'tm_ci':ci(seeds.tm),'field_nrmse':seeds.field.mean(),'field_ci':ci(seeds.field),'accepted_fraction':seeds.accepted.mean(),'accepted_ci':ci(seeds.accepted),'interval_coverage':seeds.interval.mean() if seeds.interval.notna().any() else None,'mean_interval_width':g.interval_width.mean() if g.interval_width.notna().any() else None,'accepted_tm_nrmse':ac.tm_nrmse.mean() if len(ac) else None,'accepted_unsafe_risk':ac.unsafe.mean() if len(ac) else None,'accepted_interval_coverage':ac.interval_coverage.mean() if ac.interval_coverage.notna().any() else None}
 after=g[g.frame>=80];aa=after[after.accepted==1]
 su.update({'after_tm_nrmse':after.tm_nrmse.mean(),'after_acceptance':after.accepted.mean(),'after_accepted_tm':aa.tm_nrmse.mean() if len(aa) else None,'after_accepted_unsafe':aa.unsafe.mean() if len(aa) else None,'after_interval_coverage':after.interval_coverage.mean() if after.interval_coverage.notna().any() else None})
 summary.append(su)
pd.DataFrame(summary).to_csv(D/'aggregate_results.csv',index=False)
(D/'aggregate_results.json').write_text(json.dumps(summary,indent=2))

effects=[]
for scenario in ['nominal','jump','reference_jump']:
 g=post[post.scenario==scenario].groupby(['seed','method']).tm_nrmse.mean().unstack()
 for method in ['random','no_anchor','no_inflation','periodic']:
  delta=g[method]-g.active
  rel=100*(g[method]-g.active)/g[method]
  effects.append({'scenario':scenario,'comparator':method,'paired_absolute_reduction':delta.mean(),'absolute_ci':ci(delta),'paired_relative_reduction_percent':rel.mean(),'relative_ci':ci(rel)})
(D/'paired_effects.json').write_text(json.dumps(effects,indent=2))

fault=[]
for scenario in ['mismatch','corruption','anchor_fault','loss_mismatch']:
 g=post[(post.scenario==scenario)&(post.method=='full')]
 per=[];delays=[];recover=[]
 for seed,ss in g.groupby('seed'):
  bb=ss[ss.bad_frame==1]
  per.append(bb.rejected.mean())
  flags=ss[(ss.frame>=80)&(ss.accepted==0)].frame.to_numpy()
  delays.append(int(flags[0]-80) if len(flags) else 80)
  if scenario in ['corruption','anchor_fault']:
   clean=ss[(ss.frame>=88)&(ss.accepted==1)].frame.to_numpy()
   recover.append(int(clean[0]-88) if len(clean) else 72)
 fault.append({'scenario':scenario,'fault_rejection':float(np.mean(per)),'rejection_ci':ci(per),'median_rejection_delay_frames':float(np.median(delays)),'maximum_rejection_delay_frames':max(delays),'median_recovery_delay_frames':float(np.median(recover)) if recover else None})
(D/'fault_results.json').write_text(json.dumps(fault,indent=2))

# Equal-scenario mixture for risk-availability curves. Unknown fault prevalence is not inferred.
mix=post[(post.method=='active')&post.scenario.isin(['nominal','mismatch','corruption','anchor_fault','loss_mismatch'])]
rc=[]
for threshold in np.r_[np.linspace(.8,2,31),np.geomspace(2.1,100,45)]:
 a=(mix.health<=threshold)&(mix.tm_uncertainty<=cfg['uncertainty_gate'])
 rc.append([threshold,a.mean(),mix.loc[a,'unsafe'].mean() if a.any() else np.nan,mix.loc[a,'tm_nrmse'].mean() if a.any() else np.nan])
pd.DataFrame(rc,columns=['health_threshold','accepted_fraction','unsafe_risk','accepted_tm_nrmse']).to_csv(D/'risk_availability.csv',index=False)

s=pd.read_csv(D/'sensitivity_trajectories.csv');s=s[s.frame>=12]
sens=s.groupby(['snr_db','budget','method','seed']).agg(tm=('tm_nrmse','mean'),accepted=('accepted','mean'),coverage=('interval_coverage','mean')).reset_index()
out=[]
for (snr,budget,method),g in sens.groupby(['snr_db','budget','method']):
 out.append({'snr_db':int(snr),'budget':int(budget),'method':method,'tm_nrmse':g.tm.mean(),'tm_ci':ci(g.tm),'accepted_fraction':g.accepted.mean(),'component_coverage':g.coverage.mean()})
(D/'sensitivity_summary.json').write_text(json.dumps(out,indent=2))
pd.DataFrame(out).to_csv(D/'sensitivity_summary.csv',index=False)

# First five frames after the in-model jump, including the first assimilation at frame 80.
jump=[]
for method in ['random','active','no_inflation','no_anchor','periodic']:
 g=r[(r.scenario=='jump')&(r.method==method)&r.frame.between(80,84)].groupby('seed').tm_nrmse.mean()
 jump.append({'method':method,'jump_window_tm':g.mean(),'ci':ci(g)})
(D/'jump_window.json').write_text(json.dumps(jump,indent=2))

checks=json.loads((D/'numerical_checks.json').read_text())
checks.update({'test_seed_count':int(r.seed.nunique()),'test_rows':len(r),'sensitivity_seed_count':int(s.seed.nunique()),'sensitivity_rows':len(s),'finite_primary_metrics':bool(np.isfinite(r[['tm_nrmse','field_nrmse','accepted']]).all().all()),'nominal_calibration_truth_not_in_test':not bool(set(cfg['calibration_seeds'])&set(r.seed)),'tuning_truth_not_in_test':not bool(set(cfg['tuning_seeds'])&set(r.seed)),'nonnegative_nrmse':bool((r.tm_nrmse>=0).all()),'nominal_same_fit_active_full':bool(np.allclose(r[(r.scenario=='nominal')&(r.method=='active')].tm_nrmse.to_numpy(),r[(r.scenario=='nominal')&(r.method=='full')].tm_nrmse.to_numpy()))})
mins=[]
for path in D.glob('raw_*_active.npz'):
 a=np.load(path)
 mins.append(float(np.min(np.linalg.eigvalsh(a['cov']))))
checks['minimum_retained_covariance_eigenvalue']=min(mins)
assert checks['finite_primary_metrics'] and checks['minimum_retained_covariance_eigenvalue']>0
(D/'VALIDATION_REPORT.json').write_text(json.dumps(checks,indent=2))
print(pd.DataFrame(summary)[['scenario','method','tm_nrmse','accepted_fraction','accepted_unsafe_risk','interval_coverage']].to_string(index=False))
print('Paired effects',json.dumps(effects[:4],indent=2))
print('Faults',json.dumps(fault,indent=2))
