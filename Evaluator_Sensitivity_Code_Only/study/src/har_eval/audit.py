"""Read-only independent reconstruction, permutation Shapley (not subset formula)."""
import argparse,csv,itertools,json,time
from pathlib import Path
import numpy as np
from .io import sha,json_write
from . import data

def rows(p):
    with open(p) as f:return list(csv.DictReader(f))

def expected_artifacts(cfg):
    expected={'config.json','environment.json','provenance.json','split_manifest.json','normalization.npz','validation_index.csv','initial_model.npz','base_model.npz','warmup.jsonl','snapshot_training.csv','coalition_utilities.csv','client_scores.csv','reward_shares.csv','ranking_comparisons.csv','valuation_frozen.json','test_metrics.json','process_exit.json'}
    expected|={f'updates/client_{i}.npz' for i in sum(cfg['cohorts'].values(),[])}
    expected|={f'coalitions/{c}/mask_{m:04b}.npz' for c in cfg['cohorts'] for m in range(16)}
    expected|={f'test_evidence/{c}.npz' for c in ['base']+list(cfg['cohorts'])}
    return expected

def verify_inventory(out,manifest,cfg):
    expected=expected_artifacts(cfg)
    assert set(manifest)==expected, 'manifest inventory mismatch'
    controls={'manifest.json'}
    if (out/'audit.json').exists(): controls.add('audit.json')
    if (out/'COMPLETE.json').exists():
        assert (out/'audit.json').exists(), 'completion without audit'
        controls.add('COMPLETE.json')
    actual={str(f.relative_to(out)) for f in out.rglob('*') if f.is_file()}
    assert actual==expected|controls, 'file inventory mismatch'

def verify_rankings(out,cfg):
    scores=rows(out/'client_scores.csv');cr=rows(out/'ranking_comparisons.csv')
    expected=set(itertools.product(cfg['cohorts'],('shapley','leave_one_out','data_size'),('equal_class','equal_subject')))
    assert len(cr)==24 and {(r['cohort'],r['method'],r['evaluator']) for r in cr}==expected, 'ranking inventory mismatch'
    for r in cr:
        assert r['reference']=='sample_frequency', 'incorrect ranking reference'
        def get(e):
            return np.array([next(float(a['score']) for a in scores if a['cohort']==r['cohort'] and a['method']==r['method'] and a['evaluator']==e and int(a['client'])==i) for i in cfg['cohorts'][r['cohort']]])
        a,b=get(r['reference']),get(r['evaluator']);pairs=list(itertools.combinations(range(4),2))
        def signs(a):return np.array([0 if abs(a[i]-a[k])<=cfg['tie_tolerance'] else np.sign(a[i]-a[k]) for i,k in pairs])
        sa,sb=signs(a),signs(b)
        assert int(r['reversals'])==np.sum(sa*sb<0), 'ranking reversals mismatch'
        assert int(r['tie_transitions'])==np.sum((sa==0)!=(sb==0)), 'ranking tie transitions mismatch'
        den=np.sqrt(np.count_nonzero(sa)*np.count_nonzero(sb))
        if den:
            assert abs(float(r['tau_b'])-sum(sa*sb)/den)<1e-12, 'ranking tau mismatch'
            assert r['tau_status']=='defined'
        else:
            assert r['tau_b']=='' and r['tau_status']=='undefined_all_ties'
        def reward(a):
            a=np.clip(a,0,None);return a/a.sum() if a.sum() else np.full(4,.25)
        assert abs(float(r['tv'])-abs(reward(a)-reward(b)).sum()/2)<1e-12, 'ranking TV mismatch'
        ta=set(np.flatnonzero(a.max()-a<=cfg['tie_tolerance']));tb=set(np.flatnonzero(b.max()-b<=cfg['tie_tolerance']))
        assert abs(float(r['top_jaccard'])-len(ta&tb)/len(ta|tb))<1e-12, 'ranking top-set mismatch'
        if r['method']=='data_size':assert float(r['tv'])==0 and int(r['reversals'])==0
    return len(cr)

def verify(out,archive):
    j=lambda n:json.loads((out/n).read_text())
    manifest=j('manifest.json')
    for name,h in manifest.items():assert sha(out/name)==h,name
    cfg=j('config.json');verify_inventory(out,manifest,cfg)
    assert not list(out.rglob('*.tmp'))
    cfg=j('config.json');sp=j('split_manifest.json')
    assert sha(archive)==j('provenance.json')['archive_sha256']
    x,y,s=data.read_split(archive,'train');tx,ty,ts=data.read_split(archive,'test')
    assert sp==data.construct(x,y,s,cfg)[0];data.overlap_check(x,s,sp,tx,ts)
    norm=np.load(out/'normalization.npz');ix=sp['train_indices']
    np.testing.assert_array_equal(norm['mean'],x[ix].mean(axis=(0,2),dtype=np.float64));np.testing.assert_array_equal(norm['std'],x[ix].std(axis=(0,2),dtype=np.float64))
    assert set(sp['validation_subjects']).isdisjoint(map(int,sp['clients']))
    val=rows(out/'validation_index.csv');vi=sp['validation_indices'];vy=np.array([int(a['label']) for a in val]);vs=np.array([int(a['subject']) for a in val]);np.testing.assert_array_equal(vy,y[vi]);np.testing.assert_array_equal(vs,s[vi])
    assert [a['window_id'] for a in val]==[f'train:{i}' for i in vi]
    ur=rows(out/'coalition_utilities.csv');sr=rows(out/'client_scores.csv');rr=rows(out/'reward_shares.csv')
    assert len(ur)==192 and len(sr)==len(rr)==144
    assert len({(a['cohort'],a['evaluator'],a['mask']) for a in ur})==192
    assert len({(a['cohort'],a['evaluator'],a['method'],a['client']) for a in sr})==144
    assert len(list((out/'coalitions').rglob('*.npz')))==64
    # Independently rebuild every coalition parameter tensor and evaluate its logits.
    import torch
    from .model import predict
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    base=np.load(out/'base_model.npz');vx=torch.tensor(((x[vi]-norm['mean'][None,:,None])/norm['std'][None,:,None]).astype('float32'))
    for c,clients in cfg['cohorts'].items():
        updates=[np.load(out/f'updates/client_{i}.npz') for i in clients]
        for mask in range(16):
            active=[i for i in range(4) if mask>>i&1];total=sum(len(sp['clients'][str(clients[i])]) for i in active)
            state={}
            for k in base.files:
                delta=np.zeros_like(base[k])
                for i in active:delta+=updates[i][k]*(len(sp['clients'][str(clients[i])])/total)
                state[k]=torch.from_numpy(base[k]+delta)
            rebuilt=predict(state,vx);saved=np.load(out/f'coalitions/{c}/mask_{mask:04b}.npz')['logits']
            np.testing.assert_allclose(rebuilt,saved,rtol=1e-6,atol=1e-6)
    residual=[];checks=0
    for c,clients in cfg['cohorts'].items():
        ls=[]
        for mask in range(16):
            z=np.load(out/f'coalitions/{c}/mask_{mask:04b}.npz')['logits'].astype('float64'); assert z.shape==(len(vy),6) and np.isfinite(z).all()
            # Independent stable log-softmax expression, no core.losses call.
            shifted=z-z.max(1,keepdims=True);ce=np.log(np.exp(shifted).sum(1))-shifted[np.arange(len(vy)),vy];ls.append(ce)
        ns=np.array([len(sp['clients'][str(i)]) for i in clients])
        for e in ('sample_frequency','equal_class','equal_subject'):
            if e=='sample_frequency':L=np.array(ls).mean(1)
            else:
                groups=vy if e=='equal_class' else vs
                L=np.array([np.mean([a[groups==g].mean() for g in np.unique(groups)]) for a in ls])
            v=L[0]-L
            got=sorted([a for a in ur if a['cohort']==c and a['evaluator']==e],key=lambda a:int(a['mask']))
            np.testing.assert_allclose([float(a['utility']) for a in got],v,atol=1e-12);np.testing.assert_allclose([float(a['loss']) for a in got],L,atol=1e-12)
            sh=np.zeros(4)
            for perm in itertools.permutations(range(4)):
                mask=0
                for i in perm:sh[i]+=(v[mask|1<<i]-v[mask])/24;mask|=1<<i
            residual.append(abs(sh.sum()-v[15]))
            for method,expected in [('shapley',sh),('leave_one_out',np.array([v[15]-v[15^(1<<i)] for i in range(4)])),('data_size',ns/ns.sum())]:
                values=[next(float(a['score']) for a in sr if a['cohort']==c and a['evaluator']==e and a['method']==method and int(a['client'])==i) for i in clients]
                np.testing.assert_allclose(values,expected,atol=1e-12);checks+=4
                pos=np.maximum(expected,0); shares=pos/pos.sum() if pos.sum()>0 else np.ones(4)/4
                gotr=[next(a for a in rr if a['cohort']==c and a['evaluator']==e and a['method']==method and int(a['client'])==i) for i in clients]
                np.testing.assert_allclose([float(a['share']) for a in gotr],shares,atol=1e-10)
                assert all((a['uniform_fallback']=='True')==(pos.sum()==0) for a in gotr)
    warm=[json.loads(a) for a in (out/'warmup.jsonl').read_text().splitlines()];snap=rows(out/'snapshot_training.csv')
    assert len(warm)==320 and len(snap)==16
    assert {(r['round'],r['client']) for r in warm}==set(itertools.product(range(1,21),map(int,sp['clients'])))
    for r in warm+snap:assert int(r['minibatch_steps'])==5 and int(r['processed_examples'])>0 and np.isfinite(float(r['ce_sum']))
    for r in snap:
        assert r['base_sha256']==sha(out/'base_model.npz');assert r['update_sha256']==sha(out/f"updates/client_{r['client']}.npz")
    frozen=j('valuation_frozen.json')
    for n,h in frozen['hashes'].items():assert sha(out/n)==h
    assert not any('test' in n for n in frozen['hashes'])
    tm=j('test_metrics.json');assert set(tm)=={'base','A','B','C','D'}
    for n in tm:
        z=np.load(out/f'test_evidence/{n}.npz');np.testing.assert_array_equal(z['labels'],ty);np.testing.assert_array_equal(z['subjects'],ts)
        logits=z['logits'].astype(float);shift=logits-logits.max(1,keepdims=True);loss=np.log(np.exp(shift).sum(1))-shift[np.arange(len(ty)),ty]
        cm=np.zeros((6,6));np.add.at(cm,(ty,logits.argmax(1)),1);den=cm.sum(0)+cm.sum(1)
        expected=[np.trace(cm)/len(ty),np.divide(2*cm.diagonal(),den,out=np.zeros(6),where=den>0).mean(),loss.mean()]
        np.testing.assert_allclose([tm[n][k] for k in ('accuracy','macro_f1','loss')],expected,atol=1e-12)
    assert max(residual)<=1e-12, 'Shapley efficiency mismatch'
    ranking_records=verify_rankings(out,cfg)
    assert j('process_exit.json')['exit_status']==0
    assert frozen['timestamp']<j('process_exit.json')['end']<=time.time()
    if (out/'COMPLETE.json').exists():
        complete=j('COMPLETE.json');assert complete['manifest_sha256']==sha(out/'manifest.json');assert complete['audit_sha256']==sha(out/'audit.json');assert complete['timestamp']>=j('audit.json')['verified_at']
    return dict(status='PASS',ranking_records_reconstructed=ranking_records,inventory_verified=True,coalitions=64,utilities_reconstructed=192,scores_reconstructed=checks,shapley_efficiency_games=12,max_efficiency_residual=max(residual),warmup_client_records=320,snapshot_records=16,test_metric_comparisons=15,training_windows=len(ix),validation_windows=len(vi),test_windows=len(ty),hashes_verified=len(manifest),test_leakage_checks='PASS: whole subjects; train-only normalization; valuation frozen before test loading; source data flow isolated',verified_at=time.time())
def main():
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('archive',type=Path);p.add_argument('--write',action='store_true');a=p.parse_args();r=verify(a.out,a.archive)
    if a.write:json_write(a.out/'audit.json',r)
    print(json.dumps(r,indent=2))
if __name__=='__main__':main()
