"""Descriptive analysis only; independent unit is the seed. No hypothesis tests."""
import argparse,csv,json,hashlib,itertools,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def readcsv(path):
 with path.open() as f:return list(csv.DictReader(f))
def writecsv(path,rows):
 with path.open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def interval(x,indices):
 x=np.asarray(x,dtype=float);assert x.shape==(10,) and np.isfinite(x).all()
 means=x[indices].mean(1);medians=np.median(x[indices],axis=1)
 lo,hi=np.quantile(means,[.0125,.9875]);ml,mh=np.quantile(medians,[.0125,.9875])
 return dict(n_seeds=10,mean=float(x.mean()),median=float(np.median(x)),sd=float(x.std(ddof=1)),minimum=float(x.min()),maximum=float(x.max()),mean_ci_lower=float(lo),mean_ci_upper=float(hi),median_ci_lower=float(ml),median_ci_upper=float(mh),interval_level=.975)
def selftest():
 idx=np.random.default_rng(91001).integers(0,10,size=(50000,10))
 for val in [0.,.25,1.]:
  d=interval(np.full(10,val),idx);assert d['mean_ci_lower']==d['mean_ci_upper']==val
 x=np.arange(10)/10;d=interval(x,idx);assert np.isclose(d['mean'],.45) and d['mean_ci_lower']<.45<d['mean_ci_upper']
 try:interval(np.ones(9),idx)
 except AssertionError:pass
 else:raise AssertionError('missing seed accepted')
 print('Analysis fixtures PASS')

def main():
 p=argparse.ArgumentParser();p.add_argument('--repository',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();selftest()
 audit=json.loads(a.audit.read_text());assert audit['status']=='PASS' and audit['seeds']==list(range(4101,4111));assert not (a.audit.parent/'BATCH_STOPPED.json').exists()
 seeds=list(range(4101,4111));allr=[];alls=[];allw=[];test=[];inputs={};cost=[];cohort_counts=[]
 for seed in seeds:
  run=a.repository/'results'/('pilot_4101_v2' if seed==4101 else f'seed_{seed}_study_v1_1')
  j=lambda n:json.loads((run/n).read_text())
  complete=j('COMPLETE.json');assert complete['manifest_sha256']==sha(run/'manifest.json') and complete['audit_sha256']==sha(run/'audit.json')
  for name,h in j('manifest.json').items():assert sha(run/name)==h
  assert j('audit.json')['ranking_records_reconstructed']==24 and j('audit.json')['inventory_verified']
  assert j('config.json')['seed']==seed and j('provenance.json')['commit']==audit['source']['commit']
  inputs[str(seed)]={str(f.relative_to(run)):sha(f) for f in run.rglob('*') if f.is_file()}
  allr += [dict(seed=seed,**r) for r in readcsv(run/'ranking_comparisons.csv')]
  alls += [dict(seed=seed,**r) for r in readcsv(run/'client_scores.csv')]
  allw += [dict(seed=seed,**r) for r in readcsv(run/'reward_shares.csv')]
  for model,metrics in j('test_metrics.json').items():test.append(dict(seed=seed,model=model,accuracy=metrics['accuracy'],macro_f1=metrics['macro_f1'],loss=metrics['loss']))
  cost.append(dict(seed=seed,runner_seconds_before_audit=j('process_exit.json')['elapsed_seconds'],seconds_through_complete=complete['timestamp']-j('provenance.json')['started'],raw_bytes=sum(f.stat().st_size for f in run.rglob('*') if f.is_file())))
  if seed==4101:
   split=j('split_manifest.json')
   for c,ids in split['cohorts'].items():
    for i in ids:cohort_counts.append(dict(cohort=c,subject=i,training_windows=len(split['clients'][str(i)])))
 assert len(allr)==240 and len(alls)==len(allw)==1440
 assert len({(r['seed'],r['cohort'],r['method'],r['evaluator']) for r in allr})==240
 a.out.mkdir(exist_ok=False,parents=True)
 (a.out/'input_hashes.json').write_text(json.dumps(inputs,indent=2))
 writecsv(a.out/'all_rank_comparisons.csv',allr);writecsv(a.out/'all_client_scores.csv',alls);writecsv(a.out/'all_reward_shares.csv',allw);writecsv(a.out/'cohort_counts.csv',cohort_counts);writecsv(a.out/'test_descriptive_per_seed.csv',test);writecsv(a.out/'runtime_storage.csv',cost)
 evaluators=['equal_class','equal_subject'];methods=['shapley','leave_one_out','data_size'];full_e=['sample_frequency']+evaluators
 idx=np.random.default_rng(91001).integers(0,10,size=(50000,10));np.save(a.out/'bootstrap_indices.npy',idx)
 primary=[];seedvalues=[];secondary=[]
 for e in evaluators:
  vals=[]
  for seed in seeds:
   rr=[r for r in allr if r['seed']==seed and r['evaluator']==e and r['method']=='shapley'];assert len(rr)==4
   v=np.mean([float(r['tv']) for r in rr]);vals.append(v);seedvalues.append(dict(seed=seed,contrast=e+'_vs_sample_frequency',cohort_mean_shapley_tv=float(v)))
  primary.append(dict(contrast=e+'_vs_sample_frequency',**interval(vals,idx)))
 for method,e,seed in itertools.product(methods,evaluators,seeds):
  rr=[r for r in allr if r['seed']==seed and r['evaluator']==e and r['method']==method];assert len(rr)==4
  secondary.append(dict(seed=seed,method=method,evaluator=e,mean_tv=float(np.mean([float(r['tv']) for r in rr])),mean_reversal_fraction=float(np.mean([int(r['reversals'])/6 for r in rr])),mean_tie_transitions=float(np.mean([int(r['tie_transitions']) for r in rr])),mean_top_jaccard=float(np.mean([float(r['top_jaccard']) for r in rr]))))
 writecsv(a.out/'primary_per_seed.csv',seedvalues);writecsv(a.out/'primary_summary.csv',primary);writecsv(a.out/'secondary_per_seed.csv',secondary)
 scorediag=[];disagreements=[]
 for seed,e,m in itertools.product(seeds,full_e,methods):
  ss=[r for r in alls if r['seed']==seed and r['evaluator']==e and r['method']==m];ww=[r for r in allw if r['seed']==seed and r['evaluator']==e and r['method']==m];assert len(ss)==len(ww)==16
  scorediag.append(dict(seed=seed,evaluator=e,method=m,negative_score_fraction=float(np.mean([float(r['score'])<0 for r in ss])),uniform_fallback_cohort_fraction=float(np.mean([r['uniform_fallback']=='True' for r in ww]))))
  if m!='shapley':
   distances=[]
   for c in 'ABCD':
    get=lambda meth:{int(r['client']):float(r['share']) for r in allw if r['seed']==seed and r['evaluator']==e and r['method']==meth and r['cohort']==c}
    aa,bb=get('shapley'),get(m);assert aa.keys()==bb.keys();distances.append(sum(abs(aa[i]-bb[i]) for i in aa)/2)
   disagreements.append(dict(seed=seed,evaluator=e,reference=m,mean_allocation_tv_vs_shapley=float(np.mean(distances))))
 writecsv(a.out/'score_diagnostics_per_seed.csv',scorediag);writecsv(a.out/'scoring_reference_disagreement_per_seed.csv',disagreements)
 # Seed-level descriptive summaries; no client/window/coalition pseudo-replication.
 desc=[]
 for e,m in itertools.product(full_e,methods):
  rr=[r for r in scorediag if r['evaluator']==e and r['method']==m]
  desc.append(dict(evaluator=e,method=m,n_seeds=10,mean_negative_fraction=float(np.mean([r['negative_score_fraction'] for r in rr])),mean_uniform_fallback_fraction=float(np.mean([r['uniform_fallback_cohort_fraction'] for r in rr]))))
 writecsv(a.out/'score_diagnostics_summary.csv',desc)
 ts=[]
 for model in ['base','A','B','C','D']:
  for metric in ['accuracy','macro_f1','loss']:
   v=np.array([r[metric] for r in test if r['model']==model]);assert len(v)==10
   ts.append(dict(model=model,metric=metric,n_seeds=10,mean=float(v.mean()),sd=float(v.std(ddof=1)),minimum=float(v.min()),maximum=float(v.max())))
 writecsv(a.out/'test_descriptive_summary.csv',ts)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 def save(fig,name):
  fig.savefig(a.out/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(a.out/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,4.5))
 for k,e in enumerate(evaluators):
  vals=[r['cohort_mean_shapley_tv'] for r in seedvalues if r['contrast']==e+'_vs_sample_frequency'];d=primary[k]
  ax.scatter(k+np.linspace(-.12,.12,10),vals,s=28,color=['#2274A5','#C06014'][k],label='Ten seed values' if k==0 else None)
  ax.errorbar(k+.24,d['mean'],yerr=[[d['mean']-d['mean_ci_lower']],[d['mean_ci_upper']-d['mean']]],fmt='D',color='black',capsize=5,label='Mean; 97.5% seed-bootstrap interval' if k==0 else None)
 ax.set_xticks([0,1],['Equal-class vs frequency','Equal-subject vs frequency']);ax.set_ylabel('Mean allocation TV across four fixed cohorts');ax.set_ylim(bottom=0);ax.legend(fontsize=8);ax.set_title('Prespecified evaluator contrasts · 10 independent seeds');save(fig,'primary_seed_intervals')
 fig,axes=plt.subplots(1,2,figsize=(9,3.8),sharey=True)
 for ax,e in zip(axes,evaluators):
  mat=np.array([[np.mean([int(r['reversals'])/6 for r in allr if r['evaluator']==e and r['method']==m and r['cohort']==c]) for c in 'ABCD'] for m in methods])
  im=ax.imshow(mat,vmin=0,vmax=1,cmap='Blues',aspect='auto');ax.set_xticks(range(4),list('ABCD'));ax.set_yticks(range(3),['Shapley','Leave-one-out','Data-size']);ax.set_title(e.replace('_',' ')+' vs frequency');ax.set_xlabel('Fixed cohort')
  for i in range(3):
   for k in range(4):ax.text(k,i,f'{mat[i,k]:.2f}',ha='center',va='center',color='white' if mat[i,k]>.5 else 'black')
 fig.colorbar(im,ax=axes,label='Mean fraction of six pairs reversed across seeds',shrink=.8);save(fig,'rank_reversal_descriptive')
 fig,ax=plt.subplots(figsize=(8,4))
 for k,(e,m) in enumerate(itertools.product(full_e,['leave_one_out','data_size'])):
  v=[r['mean_allocation_tv_vs_shapley'] for r in disagreements if r['evaluator']==e and r['reference']==m]
  ax.scatter(k+np.linspace(-.12,.12,10),v,s=22,color='#2274A5' if m=='leave_one_out' else '#C06014');ax.plot([k-.18,k+.18],[np.mean(v)]*2,color='black')
 ax.set_xticks(range(6),[e.replace('_',' ')+'\n'+m.replace('_',' ') for e,m in itertools.product(full_e,['leave_one_out','data_size'])],fontsize=8);ax.set_ylabel('Allocation TV vs exact Shapley; cohort mean');ax.set_ylim(bottom=0);ax.set_title('Exploratory scoring-reference comparison · points are seeds');save(fig,'scoring_reference_disagreement')
 (a.out/'analysis_specification.json').write_text(json.dumps(dict(independent_unit='seed',n_seeds=10,cohorts='four fixed repeated blocks averaged within seed',primary='two Shapley allocation TV evaluator contrasts',bootstrap_draws=50000,bootstrap_seed=91001,interval='percentile',interval_level=.975,quantiles=[.0125,.9875],mean_and_median_intervals=True,p_values_computed=False,other_metrics='exploratory descriptive only',source_script_sha256=sha(Path(__file__)),numpy=np.__version__,matplotlib=matplotlib.__version__,cross_seed_audit_sha256=sha(a.audit)),indent=2))
 print(json.dumps(primary,indent=2))
if __name__=='__main__':main()
