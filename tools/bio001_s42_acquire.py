from __future__ import annotations
import hashlib, json, urllib.request
from pathlib import Path
URL='https://osf.io/download/evhrg/'
FILENAME='AML310_moving.tar.gz'
EXPECTED_BYTES=348_444_164
EXPECTED_SHA256='144126ee9a49d311c3393deea434e1a0963d55de35318e25d98d48f9c175250a'
def sha256(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
root=Path('s42_artifacts'); raw=root/'raw'; raw.mkdir(parents=True,exist_ok=True)
target=raw/FILENAME
urllib.request.urlretrieve(URL,target)
obs=(target.stat().st_size,sha256(target))
if obs!=(EXPECTED_BYTES,EXPECTED_SHA256): raise RuntimeError(obs)
manifest={'schema':'bio-001-s42-source-manifest-v1','target':{'filename':FILENAME,'url':URL,'bytes':obs[0],'sha256':obs[1]},
'development_source_accessed':False,'sealed_holdout_requested':False,'sealed_holdout_listed':False,'sealed_holdout_downloaded':False,'sealed_holdout_opened':False}
(root/'source_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True),encoding='utf-8')
print(json.dumps({'status':'PASS','bytes':obs[0],'sha256':obs[1],'holdout_opened':False}))
