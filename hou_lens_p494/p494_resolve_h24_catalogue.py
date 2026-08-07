#!/usr/bin/env python3
import csv, hashlib, io, json, math, os, pathlib, sys
import requests

TAP=os.environ.get('ESO_TAP_CAT','https://archive.eso.org/tap_cat/sync')
TABLE='KiDS_DR5_0_ugriZYJHKs_cat_fits'
OUT=pathlib.Path('p494_tap_artifact'); OUT.mkdir(exist_ok=True)
RADIUS_DEG=15.0/3600.0; BIND_MAX_ARCSEC=5.0
H24=[
('H24GOLD-CXCOJ100201+020330','KIDS_150.1_2.2',150.506300,2.058200),
('H24GOLD-J0907+0003','KIDS_137.0_0.5',136.793700,0.055900),
('H24GOLD-J1037+0018','KIDS_159.0_0.5',159.366500,0.305700),
('H24GOLD-J1233-0227','KIDS_188.5_-2.5',188.421900,-2.460400),
('H24GOLD-J1335+0118','KIDS_204.0_1.5',203.895000,1.301500),
('H24GOLD-KIDS1042+0023','KIDS_161.0_0.5',160.655300,0.383900)]

def h(b): return hashlib.sha256(b).hexdigest()
def rows(body): return list(csv.DictReader(io.StringIO(body.decode('utf-8-sig'))))
def tap(q,stem):
 qb=q.encode(); (OUT/f'{stem}.adql').write_bytes(qb)
 r=requests.post(TAP,data={'REQUEST':'doQuery','LANG':'ADQL','FORMAT':'csv','QUERY':q},timeout=120)
 body=r.content; (OUT/f'{stem}.response').write_bytes(body)
 rec={'endpoint':TAP,'http_status':r.status_code,'query_sha256':h(qb),'response_sha256':h(body),'response_bytes':len(body)}
 (OUT/f'{stem}.receipt.json').write_text(json.dumps(rec,sort_keys=True,indent=2)+'\n')
 r.raise_for_status(); return body,rec
def sep(ra1,dec1,ra2,dec2):
 r1,r2,d1,d2=map(math.radians,[ra1,ra2,dec1,dec2]); x=math.sin((d2-d1)/2)**2+math.cos(d1)*math.cos(d2)*math.sin((r2-r1)/2)**2
 return math.degrees(2*math.asin(min(1,math.sqrt(x))))*3600

def main():
 schema_q=f"SELECT column_name, datatype, unit, description FROM TAP_SCHEMA.columns WHERE table_name='{TABLE}'"
 sb,srec=tap(schema_q,'00_schema'); sr=rows(sb)
 qa=[r for r in sr if any(k in (r.get('column_name') or '').lower() for k in ('psf','fwhm','limmag','maglim','depth'))]
 (OUT/'00_qa_column_candidates.json').write_text(json.dumps(qa,sort_keys=True,indent=2)+'\n')
 resolved=[]; failed=[]
 for i,(gid,tile,ra,dec) in enumerate(H24,1):
  q=f"""SELECT TOP 50 ID AS source_id, KIDS_TILE AS tile_id, RAJ2000 AS ra_deg, DECJ2000 AS dec_deg, MAG_AUTO-EXTINCTION_r AS r_magnitude, A_WORLD*3600.0 AS angular_size, S_ELLIPTICITY AS morphology_proxy, Flag AS extraction_flag, IMAFLAGS_ISO AS image_flag, SG2DPHOT AS star_classifier FROM {TABLE} WHERE KIDS_TILE='{tile}' AND 1=CONTAINS(POINT('ICRS',RAJ2000,DECJ2000),CIRCLE('ICRS',{ra:.9f},{dec:.9f},{RADIUS_DEG:.12f})) AND MAG_AUTO IS NOT NULL AND EXTINCTION_r IS NOT NULL AND A_WORLD>0 AND S_ELLIPTICITY IS NOT NULL"""
  body,rec=tap(q,f'{i:02d}_{gid.replace("+","p").replace("-","m")}')
  cs=[]
  for r in rows(body):
   try: s=sep(ra,dec,float(r['ra_deg']),float(r['dec_deg']))
   except Exception: continue
   r=dict(r); r['separation_arcsec']=s; cs.append(r)
  cs.sort(key=lambda z:(round(float(z['separation_arcsec']),12),z['source_id']))
  e=[z for z in cs if float(z['separation_arcsec'])<=BIND_MAX_ARCSEC]
  item={'group_id':gid,'tile_id':tile,'reference_ra_deg':ra,'reference_dec_deg':dec,'rows_returned':len(cs),'within_5arcsec':len(e),'query_receipt':rec}
  if e:
   item['selected_core_catalogue_row']=e[0]; item['binding_rule']='nearest row within 5 arcsec in exact frozen tile; 1e-12 arcsec then source_id tie-break'; resolved.append(item)
  else: item['failure']='NO_CATALOGUE_ROW_WITHIN_5_ARCSEC'; failed.append(item)
 status='PASS_H24_CORE_CATALOGUE_BINDING' if len(resolved)==6 else 'FAIL_H24_CORE_CATALOGUE_BINDING'
 m={'schema_version':1,'protocol':'P4.9.4-public-metadata-relay','claim_boundary':'Public metadata only for already-published/exposed H24 systems; no image access, model score, private control identities, or unknown targets.','catalogue_table':TABLE,'status':status,'resolved':resolved,'failures':failed,'qa_column_candidates':qa,'schema_receipt':srec}
 b=(json.dumps(m,sort_keys=True,indent=2)+'\n').encode(); (OUT/'P494_H24_CATALOGUE_BINDING.json').write_bytes(b)
 (OUT/'SHA256SUMS.txt').write_text(''.join(f'{h(p.read_bytes())}  {p.name}\n' for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='SHA256SUMS.txt'))
 print(json.dumps({'status':status,'resolved':len(resolved),'failed':len(failed),'manifest_sha256':h(b)},sort_keys=True))
 return 0 if len(resolved)==6 else 2
if __name__=='__main__': sys.exit(main())
