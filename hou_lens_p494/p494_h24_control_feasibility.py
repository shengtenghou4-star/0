#!/usr/bin/env python3
import hashlib,json,math,pathlib,sys,requests
TAP='https://archive.eso.org/tap_cat/sync'; TABLE='KiDS_DR5_0_ugriZYJHKs_cat_fits'
OUT=pathlib.Path('p494_h24_control_feasibility_artifact'); OUT.mkdir(exist_ok=True)
H24=[
('H24GOLD-CXCOJ100201+020330','KIDS_150.1_2.2',150.506300,2.058200),
('H24GOLD-J0907+0003','KIDS_137.0_0.5',136.793700,0.055900),
('H24GOLD-J1037+0018','KIDS_159.0_0.5',159.366500,0.305700),
('H24GOLD-J1233-0227','KIDS_188.5_-2.5',188.421900,-2.460400),
('H24GOLD-J1335+0118','KIDS_204.0_1.5',203.895000,1.301500),
('H24GOLD-KIDS1042+0023','KIDS_161.0_0.5',160.655300,0.383900)]
RMAG=.35; LOGSIZE=.20; MORPH=.25

def h(b): return hashlib.sha256(b).hexdigest()
def rows(body):
 o=json.loads(body.decode('utf-8-sig'))
 if isinstance(o,list): return o if not o or isinstance(o[0],dict) else []
 d=o.get('data') or o.get('rows') or o.get('results') or []; m=o.get('metadata') or o.get('columns') or []
 if d and isinstance(d[0],dict): return d
 if d and isinstance(d[0],list):
  names=[(x.get('name') or x.get('column_name') or x.get('label')) if isinstance(x,dict) else str(x) for x in m]
  return [dict(zip(names,r)) for r in d] if names and all(names) else []
 return []
def tap(q,stem):
 qb=q.encode(); (OUT/f'{stem}.adql').write_bytes(qb)
 r=requests.post(TAP,data={'REQUEST':'doQuery','LANG':'ADQL','FORMAT':'json','QUERY':q},timeout=120); b=r.content
 (OUT/f'{stem}.response.json').write_bytes(b); rec={'http_status':r.status_code,'query_sha256':h(qb),'response_sha256':h(b),'response_bytes':len(b)}
 (OUT/f'{stem}.receipt.json').write_text(json.dumps(rec,sort_keys=True,indent=2)+'\n'); r.raise_for_status(); return rows(b),rec
def sep(ra1,de1,ra2,de2):
 a,b,c,d=map(math.radians,[ra1,ra2,de1,de2]); x=math.sin((d-c)/2)**2+math.cos(c)*math.cos(d)*math.sin((b-a)/2)**2
 return math.degrees(2*math.asin(min(1,math.sqrt(x))))*3600

def bind_positive(gid,tile,ra,de,idx):
 box=15/3600; q=f"""SELECT TOP 50 ID AS source_id,KIDS_TILE AS tile_id,RAJ2000 AS ra_deg,DECJ2000 AS dec_deg,MAG_AUTO-EXTINCTION_r AS r_magnitude,A_WORLD*3600.0 AS angular_size,S_ELLIPTICITY AS morphology_proxy,Flag AS extraction_flag,IMAFLAGS_ISO AS image_flag,SG2DPHOT AS star_classifier FROM {TABLE} WHERE KIDS_TILE='{tile}' AND RAJ2000>={ra-box:.9f} AND RAJ2000<={ra+box:.9f} AND DECJ2000>={de-box:.9f} AND DECJ2000<={de+box:.9f} AND MAG_AUTO IS NOT NULL AND EXTINCTION_r IS NOT NULL AND A_WORLD>0 AND S_ELLIPTICITY IS NOT NULL"""
 rs,rec=tap(q,f'{idx:02d}_positive_binding'); cs=[]
 for r in rs:
  try: z=dict(r); z['separation_arcsec']=sep(ra,de,float(r['ra_deg']),float(r['dec_deg'])); cs.append(z)
  except: pass
 cs.sort(key=lambda z:(round(float(z['separation_arcsec']),12),str(z['source_id']))); e=[z for z in cs if float(z['separation_arcsec'])<=5]
 if not e: raise RuntimeError(f'positive binding failed {gid}')
 return e[0],rec

def main():
 groups=[]; min_count=10**9
 for i,(gid,tile,ra,de) in enumerate(H24,1):
  p,prec=bind_positive(gid,tile,ra,de,i); r=float(p['r_magnitude']); s=float(p['angular_size']); m=float(p['morphology_proxy'])
  slo=s/(10**LOGSIZE); shi=s*(10**LOGSIZE); mlo=max(0,m-MORPH); mhi=min(1,m+MORPH)
  q=f"""SELECT TOP 5000 ID AS source_id,KIDS_TILE AS tile_id,RAJ2000 AS ra_deg,DECJ2000 AS dec_deg,MAG_AUTO-EXTINCTION_r AS r_magnitude,A_WORLD*3600.0 AS angular_size,S_ELLIPTICITY AS morphology_proxy,Flag AS extraction_flag,IMAFLAGS_ISO AS image_flag,SG2DPHOT AS star_classifier FROM {TABLE} WHERE KIDS_TILE='{tile}' AND MAG_AUTO-EXTINCTION_r BETWEEN {r-RMAG:.12f} AND {r+RMAG:.12f} AND A_WORLD*3600.0 BETWEEN {slo:.12f} AND {shi:.12f} AND S_ELLIPTICITY BETWEEN {mlo:.12f} AND {mhi:.12f} AND Flag=0 AND IMAFLAGS_ISO=0 AND (SG2DPHOT IS NULL OR SG2DPHOT<>1) AND MAG_AUTO IS NOT NULL AND EXTINCTION_r IS NOT NULL AND A_WORLD>0 AND S_ELLIPTICITY IS NOT NULL"""
  rs,rec=tap(q,f'{i:02d}_candidate_pool'); clean=[z for z in rs if str(z.get('source_id'))!=str(p['source_id'])]
  # Preliminary normalized score uses only the three object-varying components. The two tile-QA components are proven identically zero under exact tile equality.
  for z in clean:
   dr=(float(z['r_magnitude'])-r)/RMAG; ds=(math.log10(float(z['angular_size']))-math.log10(s))/LOGSIZE; dm=(float(z['morphology_proxy'])-m)/MORPH
   z['preliminary_three_component_distance']=math.sqrt(dr*dr+ds*ds+dm*dm)
  clean.sort(key=lambda z:(round(float(z['preliminary_three_component_distance']),12),str(z['source_id'])))
  (OUT/f'{i:02d}_{gid.replace("+","p").replace("-","m")}_candidates.json').write_text(json.dumps(clean,sort_keys=True,indent=2)+'\n')
  min_count=min(min_count,len(clean)); groups.append({'group_id':gid,'tile_id':tile,'positive_catalogue_row':p,'positive_binding_receipt':prec,'candidate_query_receipt':rec,'preliminary_candidate_count':len(clean),'first_ten_source_ids':[z['source_id'] for z in clean[:10]],'three_component_distance_only_due_exact_tile_qa_invariance':True})
 status='PASS_H24_METADATA_FEASIBILITY_ALL_GROUPS_HAVE_AT_LEAST_3_PRELIMINARY_CANDIDATES' if min_count>=3 else 'FAIL_H24_METADATA_FEASIBILITY_DEFICIENT_GROUP'
 m={'schema_version':1,'protocol':'P4.9.4-h24-control-feasibility','status':status,'claim_boundary':'Metadata feasibility only for six already-public H24 development systems. No control identity freeze. Full known-lens/forbidden-identity exclusion and global 11-group atomic assignment remain mandatory before any identity can become authoritative.','frozen_calipers':{'r_magnitude':RMAG,'log10_angular_size':LOGSIZE,'morphology_proxy':MORPH,'local_r_limiting_magnitude':.30,'local_r_psf_fwhm':.10},'tile_quality_invariance':'Exact tile equality makes both tile-level quality differences zero for every same-tile candidate, so preliminary ranking is exactly the three object-varying components before forbidden-identity exclusions.','groups':groups,'minimum_preliminary_candidate_count':min_count}
 b=(json.dumps(m,sort_keys=True,indent=2)+'\n').encode(); (OUT/'P494_H24_CONTROL_FEASIBILITY.json').write_bytes(b); (OUT/'SHA256SUMS.txt').write_text(''.join(f'{h(p.read_bytes())}  {p.name}\n' for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='SHA256SUMS.txt'))
 print(json.dumps({'status':status,'min_count':min_count,'counts':[g['preliminary_candidate_count'] for g in groups],'manifest_sha256':h(b)},sort_keys=True)); return 0 if min_count>=3 else 2
if __name__=='__main__': sys.exit(main())
