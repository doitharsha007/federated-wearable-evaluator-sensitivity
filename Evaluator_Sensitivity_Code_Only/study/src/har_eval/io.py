import os,json,hashlib,tempfile
from pathlib import Path
import numpy as np

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def atomic(path,writer):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise FileExistsError(path)
    fd,tmp=tempfile.mkstemp(prefix='.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f: writer(f); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
        d=os.open(path.parent,os.O_RDONLY)
        try: os.fsync(d)
        finally: os.close(d)
    except BaseException:
        # Preserve any incomplete temporary evidence.
        raise

def json_write(path,obj): atomic(path,lambda f:f.write(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False).encode()))
def npz_write(path,**obj): atomic(path,lambda f:np.savez_compressed(f,**obj))
def csv_write(path,rows):
    import csv,io
    assert rows
    s=io.StringIO(); w=csv.DictWriter(s,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    atomic(path,lambda f:f.write(s.getvalue().encode()))
def stream(seed,*parts): return int.from_bytes(hashlib.sha256('|'.join(map(str,(seed,)+parts)).encode()).digest()[:8],'little')%(2**32)
