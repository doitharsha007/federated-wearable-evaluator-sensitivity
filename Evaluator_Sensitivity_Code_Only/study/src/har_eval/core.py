"""Pure evaluator and fixed-update game functions. No training/test access."""
import itertools, math
import numpy as np
from scipy.special import logsumexp
from scipy.stats import kendalltau

def losses(logits, labels):
    z=np.asarray(logits,dtype=np.float64); y=np.asarray(labels,dtype=int)
    assert z.shape==(len(y),6) and np.isfinite(z).all()
    return logsumexp(z,axis=1)-z[np.arange(len(y)),y]

def weights(labels,subjects,kind):
    y=np.asarray(labels); s=np.asarray(subjects)
    if kind=='sample_frequency': return np.ones(len(y))/len(y)
    key=y if kind=='equal_class' else s
    assert kind in ('equal_class','equal_subject')
    groups,cnt=np.unique(key,return_counts=True)
    if kind=='equal_class': assert np.array_equal(groups,np.arange(6))
    return np.array([1/(len(groups)*cnt[np.where(groups==k)[0][0]]) for k in key])

def aggregate(base,updates,counts,mask):
    ids=[i for i in range(len(updates)) if mask>>i&1]
    if not ids: return {k:v.clone() for k,v in base.items()}
    total=sum(counts[i] for i in ids)
    return {k:base[k]+sum((updates[i][k]*(counts[i]/total) for i in ids)) for k in base}

def shapley(v):
    v=np.asarray(v,dtype=float); assert v.shape==(16,) and np.isfinite(v).all()
    return np.array([sum(math.factorial(s.bit_count())*math.factorial(3-s.bit_count())/24*(v[s|1<<i]-v[s]) for s in range(16) if not s>>i&1) for i in range(4)])

def loo(v): return np.array([v[15]-v[15^(1<<i)] for i in range(4)])
def data_size(counts): return np.asarray(counts)/sum(counts)
def rewards(scores):
    p=np.maximum(scores,0); fallback=bool(p.sum()==0)
    return (np.ones(4)/4 if fallback else p/p.sum()),fallback

def compare(a,b,tol=1e-8):
    a=np.asarray(a); b=np.asarray(b)
    signs=lambda x: np.array([0 if abs(x[i]-x[j])<=tol else np.sign(x[i]-x[j]) for i,j in itertools.combinations(range(4),2)])
    sa,sb=signs(a),signs(b)
    # Tau-b from the same tolerance-aware pair signs used for reversals.
    den=np.sqrt(np.count_nonzero(sa)*np.count_nonzero(sb))
    tau=float(np.sum(sa*sb)/den) if den else None
    topa=set(np.flatnonzero(a.max()-a<=tol)); topb=set(np.flatnonzero(b.max()-b<=tol))
    ra,_=rewards(a); rb,_=rewards(b)
    return dict(tv=float(abs(ra-rb).sum()/2),reversals=int(np.sum(sa*sb<0)),tie_transitions=int(np.sum((sa==0)!=(sb==0))),tau_b=tau,tau_status='defined' if den else 'undefined_all_ties',top_jaccard=len(topa&topb)/len(topa|topb))
