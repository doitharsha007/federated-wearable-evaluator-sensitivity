"""Faults use isolated copies of the pilot, never alter source evidence or train."""
import csv,json,os,shutil,subprocess
from pathlib import Path
import pytest
from har_eval.run import finalize
from har_eval.io import sha

@pytest.fixture
def prepared(tmp_path):
    origin=Path(os.environ['HAR_PILOT_FIXTURE'])
    out=tmp_path/'candidate';shutil.copytree(origin,out)
    (out/'COMPLETE.json').unlink();(out/'audit.json').unlink()
    return out

def rehash(out,name):
    # Keep hashes consistent to prove semantic audits reject the injected fault.
    for filename in ['valuation_frozen.json','manifest.json']:
        p=out/filename;obj=json.loads(p.read_text());hashes=obj.get('hashes',obj)
        if name in hashes: hashes[name]=sha(out/name)
        p.write_text(json.dumps(obj))
    p=out/'manifest.json';m=json.loads(p.read_text());m['valuation_frozen.json']=sha(out/'valuation_frozen.json');p.write_text(json.dumps(m))

@pytest.mark.parametrize('fault',['ranking','missing_file','unexpected_file','missing_manifest_entry'])
def test_fault_prevents_completion(prepared,fault,capfd):
    out=prepared
    if fault=='ranking':
        p=out/'ranking_comparisons.csv'
        with p.open() as f:rr=list(csv.DictReader(f))
        rr[0]['reversals']='99'
        with p.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
        rehash(out,p.name)
    elif fault=='missing_file':(out/'snapshot_training.csv').unlink()
    elif fault=='unexpected_file':(out/'unexpected.txt').write_text('injected')
    else:
        p=out/'manifest.json';m=json.loads(p.read_text());m.pop('snapshot_training.csv');p.write_text(json.dumps(m))
    with pytest.raises(subprocess.CalledProcessError):finalize(out,Path('data/uci_har.zip'))
    evidence=capfd.readouterr()
    expected={'ranking':'ranking reversals mismatch','missing_file':'FileNotFoundError','unexpected_file':'file inventory mismatch','missing_manifest_entry':'manifest inventory mismatch'}[fault]
    assert expected in evidence.err
    (out.parent/'expected_failure.txt').write_text(evidence.err)
    assert not (out/'COMPLETE.json').exists()
    assert not (out/'audit.json').exists()

def test_valid_artifacts_complete_after_checks(prepared):
    finalize(prepared,Path('data/uci_har.zip'))
    audit=json.loads((prepared/'audit.json').read_text());marker=json.loads((prepared/'COMPLETE.json').read_text())
    assert audit['ranking_records_reconstructed']==24 and audit['inventory_verified']
    assert audit['verified_at']<=marker['timestamp']
    assert marker['audit_sha256']==sha(prepared/'audit.json')
