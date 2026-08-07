from __future__ import annotations
import hashlib,json,sys,zipfile
from pathlib import Path

p=Path(sys.argv[1]); out=Path(sys.argv[2])
h=hashlib.sha256()
with p.open('rb') as f:
    for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
with zipfile.ZipFile(p,'r') as z:
    names=sorted(z.namelist())
    included=[]
    for name in names:
        parts=[q.lower() for q in Path(name).parts]
        gait=None
        if 'n2_a1_crawling' in parts: gait='crawl'
        if 'n2_a1_swimming' in parts: gait='swim'
        if gait and name.lower().endswith('.txt'):
            info=z.getinfo(name)
            included.append({'gait':gait,'path':name,'uncompressed_bytes':int(info.file_size),'compressed_bytes':int(info.compress_size),'crc32':f'{info.CRC:08x}'})
manifest={'schema':'bio-001-s48-archive-manifest-v1','filename':p.name,'bytes':p.stat().st_size,'sha256':h.hexdigest(),'zip_members':len(names),'included_txt_metadata':included,'included_crawl_txt':sum(q['gait']=='crawl' for q in included),'included_swim_txt':sum(q['gait']=='swim' for q in included),'numeric_members_extracted':False,'numeric_members_opened':False}
out.write_text(json.dumps(manifest,indent=2,sort_keys=True),encoding='utf-8')
Path(str(out)+'.names.txt').write_text('\n'.join(names)+'\n',encoding='utf-8')
print(json.dumps({k:manifest[k] for k in ('bytes','sha256','zip_members','included_crawl_txt','included_swim_txt','numeric_members_extracted','numeric_members_opened')},sort_keys=True))
