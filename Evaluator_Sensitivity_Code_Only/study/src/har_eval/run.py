"""One authorized seed per invocation; test is opened only after valuation freeze."""
import argparse,json,platform,subprocess,time,traceback,sys
from pathlib import Path
import numpy as np,torch,yaml
from . import core,data
from .model import Model,fit,predict
from .io import *
EVALUATORS=['sample_frequency','equal_class','equal_subject']
def save_state(path,state): npz_write(path,**{k:v.numpy() for k,v in state.items()})
def metrics(z,y):
    ce=core.losses(z,y);pred=z.argmax(1); cm=np.zeros((6,6),dtype=int);np.add.at(cm,(y,pred),1)
    den=cm.sum(0)+cm.sum(1);f1=np.divide(2*cm.diagonal(),den,out=np.zeros(6),where=den>0)
    return dict(accuracy=float(np.trace(cm)/len(y)),macro_f1=float(f1.mean()),loss=float(ce.mean()),loss_sum=float(ce.sum()),count=len(y),confusion=cm.tolist())
def execute(cfg,seed,archive,out):
    start=time.time();torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(stream(seed,'init'))
    json_write(out/'config.json',dict(cfg,seed=seed));json_write(out/'environment.json',dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),threads=1))
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip(),'dirty source'
    json_write(out/'provenance.json',dict(commit=commit,archive_sha256=sha(archive),archive_source='https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip',started=start))
    x,y,s=data.read_split(archive,'train');split,mean,std=data.construct(x,y,s,cfg);data.overlap_check(x,s,split)
    json_write(out/'split_manifest.json',split);npz_write(out/'normalization.npz',mean=mean,std=std)
    norm=lambda a:torch.tensor(((a-mean[None,:,None])/std[None,:,None]).astype('float32'))
    xx=norm(x);yy=torch.tensor(y);vi=split['validation_indices'];vx=xx[vi]
    csv_write(out/'validation_index.csv',[dict(window_id=f'train:{i}',label=int(y[i]),subject=int(s[i])) for i in vi])
    state=Model().state_dict();save_state(out/'initial_model.npz',state);ids=list(split['clients']);counts=[len(split['clients'][i]) for i in ids]
    warm=[]
    for r in range(1,cfg['warmup_rounds']+1):
        updates=[]
        for i in ids:
            ix=split['clients'][i];local,rec=fit(state,xx[ix],yy[ix],stream(seed,'warmup',r,i),cfg)
            updates.append({k:local[k]-state[k] for k in state});warm.append(dict(round=r,client=int(i),**rec))
        state=core.aggregate(state,updates,counts,(1<<16)-1)
        print(f'warmup {r}/20',flush=True)
    atomic(out/'warmup.jsonl',lambda f:f.write(''.join(json.dumps(a)+'\n' for a in warm).encode()))
    save_state(out/'base_model.npz',state);updates={};snap=[]
    for i in ids:
        ix=split['clients'][i];local,rec=fit(state,xx[ix],yy[ix],stream(seed,'snapshot',i),cfg)
        updates[i]={k:local[k]-state[k] for k in state};p=out/f'updates/client_{i}.npz';save_state(p,updates[i]);snap.append(dict(client=int(i),count=len(ix),base_sha256=sha(out/'base_model.npz'),update_sha256=sha(p),**rec))
    csv_write(out/'snapshot_training.csv',snap)
    util=[];scores=[];reward=[];comparisons=[];full={}
    for cohort,subjects in cfg['cohorts'].items():
        up=[updates[str(i)] for i in subjects];ns=[len(split['clients'][str(i)]) for i in subjects];ll=[]
        for mask in range(16):
            st=core.aggregate(state,up,ns,mask);z=predict(st,vx)
            npz_write(out/f'coalitions/{cohort}/mask_{mask:04b}.npz',logits=z)
            ll.append(core.losses(z,y[vi]))
            if mask==15: full[cohort]=st
        scoremap={}
        for e in EVALUATORS:
            w=core.weights(y[vi],s[vi],e);ls=np.array(ll)@w;v=ls[0]-ls
            for mask in range(16): util.append(dict(cohort=cohort,evaluator=e,mask=mask,loss=float(ls[mask]),utility=float(v[mask])))
            for method,sc in [('shapley',core.shapley(v)),('leave_one_out',core.loo(v)),('data_size',core.data_size(ns))]:
                rr,fb=core.rewards(sc);scoremap[e,method]=sc
                for j,i in enumerate(subjects):
                    scores.append(dict(cohort=cohort,evaluator=e,method=method,client=i,score=float(sc[j])))
                    reward.append(dict(cohort=cohort,evaluator=e,method=method,client=i,share=float(rr[j]),uniform_fallback=fb))
        for method in ('shapley','leave_one_out','data_size'):
            for e in EVALUATORS[1:]: comparisons.append(dict(cohort=cohort,method=method,reference='sample_frequency',evaluator=e,**core.compare(scoremap['sample_frequency',method],scoremap[e,method])))
    csv_write(out/'coalition_utilities.csv',util);csv_write(out/'client_scores.csv',scores);csv_write(out/'reward_shares.csv',reward);csv_write(out/'ranking_comparisons.csv',comparisons)
    # This immutable freeze exists before any test split is loaded or evaluated.
    files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
    json_write(out/'valuation_frozen.json',dict(timestamp=time.time(),hashes=files))
    tx,ty,ts=data.read_split(archive,'test');data.overlap_check(x,s,split,tx,ts)
    txx=norm(tx);tm={}
    for name,st in [('base',state)]+list(full.items()):
        z=predict(st,txx);npz_write(out/f'test_evidence/{name}.npz',logits=z,labels=ty,subjects=ts);tm[name]=metrics(z,ty)
    json_write(out/'test_metrics.json',tm)
    json_write(out/'process_exit.json',dict(exit_status=0,exception=None,start=start,end=time.time(),elapsed_seconds=time.time()-start))
    manifest={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
    json_write(out/'manifest.json',manifest)
    finalize(out,archive)

def finalize(out,archive):
    # Fresh verifier process, then completion marker LAST.
    subprocess.run([sys.executable,'-m','har_eval.audit',str(out),str(archive),'--write'],check=True)
    json_write(out/'COMPLETE.json',dict(manifest_sha256=sha(out/'manifest.json'),audit_sha256=sha(out/'audit.json'),timestamp=time.time()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--archive',type=Path,default=Path('data/uci_har.zip'));a=p.parse_args()
    cfg=yaml.safe_load(Path('configs/study_v1.yaml').read_text());assert a.seed in cfg['seeds']
    a.out.mkdir(parents=True,exist_ok=False)
    try: execute(cfg,a.seed,a.archive,a.out)
    except BaseException as ex:
        json_write(a.out/'FAILED.json',dict(exception=repr(ex),traceback=traceback.format_exc(),timestamp=time.time()));raise
if __name__=='__main__':main()
