from pathlib import Path
import csv,json,hashlib
import numpy as np
b=Path(__file__).resolve().parent;out=b/'analysis';repo=b.parent/'har-evaluator-sensitivity-v2'
def rows(p):
 with p.open() as f:return list(csv.DictReader(f))
seeds=list(range(4101,4111));primary=rows(out/'primary_summary.csv');per=rows(out/'primary_per_seed.csv');draws=np.load(out/'bootstrap_indices.npy')
assert draws.shape==(50000,10)
np.testing.assert_array_equal(draws,np.random.default_rng(91001).integers(0,10,size=(50000,10)))
for e in ['equal_class','equal_subject']:
 vals=[]
 for seed in seeds:
  run=repo/'results'/('pilot_4101_v2' if seed==4101 else f'seed_{seed}_study_v1_1');rr=rows(run/'reward_shares.csv');ds=[]
  for c in 'ABCD':
   get=lambda ev:{int(r['client']):float(r['share']) for r in rr if r['cohort']==c and r['evaluator']==ev and r['method']=='shapley'}
   aa,bb=get(e),get('sample_frequency');assert len(aa)==len(bb)==4 and aa.keys()==bb.keys()
   ds.append(sum(abs(aa[i]-bb[i]) for i in aa)/2)
  v=sum(ds)/4;vals.append(v)
  got=next(float(r['cohort_mean_shapley_tv']) for r in per if int(r['seed'])==seed and r['contrast']==e+'_vs_sample_frequency');assert abs(got-v)<1e-14
 vals=np.array(vals);means=np.array([sum(vals[d])/10 for d in draws]);med=np.median(vals[draws],axis=1)
 ref=next(r for r in primary if r['contrast']==e+'_vs_sample_frequency')
 for name,val in [('mean',vals.mean()),('median',np.median(vals)),('mean_ci_lower',np.quantile(means,.0125)),('mean_ci_upper',np.quantile(means,.9875)),('median_ci_lower',np.quantile(med,.0125)),('median_ci_upper',np.quantile(med,.9875))]:assert abs(float(ref[name])-val)<1e-14,name
# Every raw file still matches the analysis input ledger.
ledger=json.loads((out/'input_hashes.json').read_text())
for seed in seeds:
 run=repo/'results'/('pilot_4101_v2' if seed==4101 else f'seed_{seed}_study_v1_1')
 for name,h in ledger[str(seed)].items():assert hashlib.sha256((run/name).read_bytes()).hexdigest()==h
result={'status':'PASS','seed_endpoints_reconstructed_from_shares':20,'primary_summary_fields_checked':12,'bootstrap_indices_regenerated':500000,'all_analysis_input_hashes_verified':True,'independent_unit':'seed','n_seeds':10}
(b/'analysis_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
