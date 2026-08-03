#!/usr/bin/env python3
from __future__ import annotations
import gzip, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PARTS=[ROOT/'phase34_reconstruct_tier1.py.gz.part00',ROOT/'phase34_reconstruct_tier1.py.gz.part01',ROOT/'phase34_reconstruct_tier1.py.gz.part02']
GZ_SHA='d13a1a8466a472dd78595214a2073628846d778ea891ac73a58a4cee300b710f'
RAW_SHA='cc61864d37d8d011b405c2f3a2e0f8afc43555606072c7aefae5837c3b84846c'
def sha(data:bytes)->str:return hashlib.sha256(data).hexdigest()
packed=b''.join(p.read_bytes() for p in PARTS)
if sha(packed)!=GZ_SHA: raise SystemExit('packed digest mismatch')
raw=gzip.decompress(packed)
if sha(raw)!=RAW_SHA: raise SystemExit('source digest mismatch')
out=ROOT/'phase34_reconstruct_tier1.py'
out.write_bytes(raw); out.chmod(0o755)
print(f'PASS {out} {len(raw)} {RAW_SHA}')
