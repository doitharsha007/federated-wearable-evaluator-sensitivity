import io,zipfile,hashlib
import numpy as np
from .io import sha
CHANNELS=[f'{kind}_{axis}' for kind in ('body_acc','body_gyro','total_acc') for axis in 'xyz']
EXPECTED=[1,3,5,6,7,8,11,14,15,16,17,19,21,22,23,25,26,27,28,29,30]

def read_split(archive,split):
    assert split in ('train','test')
    with zipfile.ZipFile(archive) as outer:
        nested=io.BytesIO(outer.read('UCI HAR Dataset.zip'))
    with zipfile.ZipFile(nested) as z:
        prefix='UCI HAR Dataset/'+split+'/'
        read=lambda name:np.loadtxt(io.BytesIO(z.read(prefix+name)))
        x=np.stack([read(f'Inertial Signals/{c}_{split}.txt') for c in CHANNELS],axis=1).astype('float32')
        y=read(f'y_{split}.txt').astype('int64')-1
        s=read(f'subject_{split}.txt').astype('int64')
    assert x.shape==(len(y),9,128) and set(y)==set(range(6))
    return x,y,s

def construct(x,y,s,cfg):
    assert sorted(np.unique(s).tolist())==EXPECTED
    assert cfg['validation_subjects']==[EXPECTED[i] for i in (0,4,8,12,16)]
    clients=sum(cfg['cohorts'].values(),[])
    assert len(clients)==len(set(clients))==16
    assert set(clients)|set(cfg['validation_subjects'])==set(EXPECTED)
    assert not set(clients)&set(cfg['validation_subjects'])
    train=np.flatnonzero(np.isin(s,clients)); val=np.flatnonzero(np.isin(s,cfg['validation_subjects']))
    assert len(train)+len(val)==len(y) and not set(train)&set(val)
    assert set(y[val])==set(range(6))
    indices={str(i):np.flatnonzero(s==i).tolist() for i in clients}
    assert sorted(sum(indices.values(),[]))==train.tolist()
    mean=x[train].mean(axis=(0,2),dtype=np.float64); std=x[train].std(axis=(0,2),dtype=np.float64)
    assert (std>0).all()
    manifest=dict(official_train_count=len(y),train_indices=train.tolist(),validation_indices=val.tolist(),clients=indices,validation_subjects=cfg['validation_subjects'],cohorts=cfg['cohorts'],normalization_fit='training_clients_only')
    return manifest,mean,std

def content_hashes(x): return [hashlib.sha256(a.tobytes()).hexdigest() for a in x]
def overlap_check(x,s,manifest,testx=None,tests=None):
    train=manifest['train_indices']; val=manifest['validation_indices']; h=content_hashes(x)
    assert not set(h[i] for i in train)&set(h[i] for i in val), 'train/validation duplicate windows'
    if testx is not None:
        assert not set(s)&set(tests), 'official subject overlap'
        assert not set(h)&set(content_hashes(testx)), 'official train/test duplicate windows'
    return True
