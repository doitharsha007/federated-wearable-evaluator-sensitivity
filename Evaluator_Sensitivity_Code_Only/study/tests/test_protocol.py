import itertools,json
from pathlib import Path
import numpy as np,pytest,torch,yaml
from har_eval import core,data
from har_eval.io import json_write,sha
CFG=yaml.safe_load(Path('configs/study_v1.yaml').read_text())
@pytest.mark.parametrize('kind',['sample_frequency','equal_class','equal_subject'])
def test_weights(kind):
    y=np.array([0,0,0,1,2,3,4,5]);s=np.array([1,1,1,1,1,7,7,7]);w=core.weights(y,s,kind)
    assert np.isclose(w.sum(),1) and (w>0).all()
    group=y if kind=='equal_class' else s
    if kind!='sample_frequency':
        for k in np.unique(group):assert np.isclose(w[group==k].sum(),1/len(np.unique(group)))
def test_missing_class():
    with pytest.raises(AssertionError):core.weights([0,1],[1,2],'equal_class')
@pytest.mark.parametrize('values',[[1,2,3,4],[0,0,0,0],[-1,0,2,7]])
def test_additive(values):
    v=[sum(values[i] for i in range(4) if mask>>i&1) for mask in range(16)]
    np.testing.assert_allclose(core.shapley(v),values,atol=1e-14);np.testing.assert_allclose(core.loo(v),values)
def test_symmetric_and_dummy():
    v=np.array([float((m&7)==7) for m in range(16)])
    np.testing.assert_allclose(core.shapley(v),[1/3,1/3,1/3,0])
def test_random_permutation_reference():
    v=np.random.default_rng(6).normal(size=16);v[0]=0;expected=np.zeros(4)
    for order in itertools.permutations(range(4)):
        mask=0
        for i in order:expected[i]+=(v[mask|1<<i]-v[mask])/24;mask|=1<<i
    np.testing.assert_allclose(core.shapley(v),expected,atol=1e-14);assert np.isclose(expected.sum(),v[-1])
def test_aggregation():
    b={'x':torch.tensor([2.])};u=[{'x':torch.tensor([float(i)])} for i in range(1,5)];n=[1,2,3,4]
    assert core.aggregate(b,u,n,0)['x']==2
    for i in range(4):assert core.aggregate(b,u,n,1<<i)['x']==3+i
    assert core.aggregate(b,u,n,15)['x']==5
@pytest.mark.parametrize('scores,expected,fb',[([-1,-2,0,-4],[.25]*4,True),([-1,1,3,0],[0,.25,.75,0],False)])
def test_rewards(scores,expected,fb):
    r,f=core.rewards(scores);np.testing.assert_allclose(r,expected);assert f==fb
def test_ranking():
    assert core.compare([1,2,3,4],[4,3,2,1])['reversals']==6
    assert core.compare([1]*4,[2]*4)['tau_b'] is None
    assert core.compare([1,1+1e-9,2,3],[1,1,2,3])['tie_transitions']==0
    np.testing.assert_allclose(core.data_size([1,2,3,4]),[.1,.2,.3,.4])
def test_loss():
    np.testing.assert_allclose(core.losses(np.zeros((6,6)),np.arange(6)),np.log(6))
def test_atomic(tmp_path):
    p=tmp_path/'a.json';json_write(p,{'a':1});assert json.loads(p.read_text())=={'a':1};assert len(sha(p))==64
    with pytest.raises(FileExistsError):json_write(p,{'a':2})
    assert not list(tmp_path.glob('*.tmp'))
def test_real_official_data():
    p=Path('data/uci_har.zip');assert p.exists(),'official archive required'
    x,y,s=data.read_split(p,'train');tx,ty,ts=data.read_split(p,'test');sp,m,sd=data.construct(x,y,s,CFG)
    assert len(y)==7352 and len(ty)==2947
    assert len(sp['clients'])==16 and all(len(v)==4 for v in CFG['cohorts'].values())
    assert sorted(sp['train_indices']+sp['validation_indices'])==list(range(7352))
    assert data.overlap_check(x,s,sp,tx,ts)
    # Poisoned held-out test data cannot affect construct: it takes train arrays only.
    before=m.copy();tx[:]=999;assert np.array_equal(before,data.construct(x,y,s,CFG)[1])
def test_missing_subject_stops():
    x=np.zeros((21,9,128),dtype='float32');s=np.array(data.EXPECTED);y=np.arange(21)%6
    with pytest.raises(AssertionError):data.construct(x[:-1],y[:-1],s[:-1],CFG)
def test_duplicate_split_detection():
    x=np.zeros((2,9,128));sp={'train_indices':[0],'validation_indices':[1]}
    with pytest.raises(AssertionError):data.overlap_check(x,np.array([1,2]),sp)

def test_incomplete_output_rejected(tmp_path):
    from har_eval.audit import verify
    with pytest.raises(FileNotFoundError):verify(tmp_path,Path('data/uci_har.zip'))
def test_hash_corruption_rejected(tmp_path):
    from har_eval.audit import verify
    json_write(tmp_path/'piece.json',{'a':1});json_write(tmp_path/'manifest.json',{'piece.json':'0'*64})
    with pytest.raises(AssertionError):verify(tmp_path,Path('data/uci_har.zip'))
def test_atomic_failure_preserved(tmp_path):
    from har_eval.io import atomic
    def fail(f):f.write(b'partial');raise RuntimeError('injected')
    with pytest.raises(RuntimeError):atomic(tmp_path/'final',fail)
    assert not (tmp_path/'final').exists() and len(list(tmp_path.glob('*.tmp')))==1
