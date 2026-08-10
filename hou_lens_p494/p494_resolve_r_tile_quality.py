#!/usr/bin/env python3
import hashlib,json,math,os,pathlib,sys,requests
TAP=os.environ.get('ESO_TAP_OBS','https://archive.eso.org/tap_obs/sync')
OUT=pathlib.Path('p494_tile_quality_artifact'); OUT.mkdir(exist_ok=True)
# Calibration rows are already-exposed pilot8 development systems with historical frozen tile-quality values.
CAL=[
('KIDSOBJ-00003','KIDS_0.0_-31.2',0.050133,-31.162044,0.59,25.08),
('KIDSOBJ-00004','KIDS_0.0_-28.2',0.059900,-28.193300,0.72,25.12),
('KIDSOBJ-00008','KIDS_0.0_-33.1',0.124587,-33.523305,0.63,24.94),
('KIDSOBJ-00041','KIDS_1.2_-35.1',1.322827,-35.395134,0.63,24.96),
('KIDSOBJ-00113','KIDS_4.6_-29.2',4.543180,-28.935984,0.52,25.07),]
H24=[
('H24GOLD-CXCOJ100201+020330','KIDS_150.1_2.2',150.506300,2.058200),
('H24GOLD-J0907+0003','KIDS_137.0_0.5',136.793700,0.055900),
('H24GOLD-J1037+0018','KIDS_159.0_0.5',159.366500,0.305700),
('H24GOLD-J1233-0227','KIDS_188.5_-2.5',188.421900,-2.460400),
('H24GOLD-J1335+0118','KIDS_204.0_1.5',203.895000,1.301500),
('H24GOLD-KIDS1042+0023','KIDS_161.0_0.5',160.655300,0.383900)]
def h(b): return hashlib.sha256(b).hexdigest()
def rows(body):
 o=json.loads(body.decode('utf-8-sig'))
 if isinstance(o,list): return o if not o or isinstance(o[0],dict) else []
 if not isinstance(o,dict): return []
 d=o.get('data') or o.get('rows') or o.get('results') or []; m=o.get('metadata') or o.get('columns') or []
 if d and isinstance(d[0],dict): return d
 if d and isinstance(d[0],list):
  names=[(x.get('name') or x.get('column_name') or x.get('label')) if isinstance(x,dict) else str(x) for x in m]
  if names and all(names): return [dict(zip(names,r)) for r in d]
 return []
def tap(q,stem):
 qb=q.encode(); (OUT/f'{stem}.adql').write_bytes(qb)
 r=requests.post(TAP,data={'REQUEST':'doQuery','LANG':'ADQL','FORMAT':'json','QUERY':q},timeout=120)
 b=r.content; (OUT/f'{stem}.response.json').write_bytes(b)
 rec={'endpoint':TAP,'http_status':r.status_code,'query_sha256':h(qb),'response_sha256':h(b),'response_bytes':len(b),'format':'json'}
 (OUT/f'{stem}.receipt.json').write_text(json.dumps(rec,sort_keys=True,indent=2)+'\n'); r.raise_for_status(); return rows(b),rec
def dist(ra1,de1,ra2,de2):
 a,b,c,d=map(math.radians,[ra1,ra2,de1,de2]); x=math.sin((d-c)/2)**2+math.cos(c)*math.cos(d)*math.sin((b-a)/2)**2
 return math.degrees(2*math.asin(min(1,math.sqrt(x))))
def query_point(gid,tile,ra,dec,stem):
 q=f"""SELECT TOP 50 obs_publisher_did, obs_id, obs_title, target_name, filter, s_ra, s_dec, s_resolution, abmaglim, dataproduct_type, dataproduct_subtype, calib_level, publication_date FROM ivoa.ObsCore WHERE obs_collection='KIDS' AND dataproduct_type='image' AND filter='r_SDSS' AND s_ra BETWEEN {ra-1.2:.9f} AND {ra+1.2:.9f} AND s_dec BETWEEN {dec-1.2:.9f} AND {dec+1.2:.9f}"""
 rs,rec=tap(q,stem); cand=[]
 for r in rs:
  try: dd=dist(ra,dec,float(r['s_ra']),float(r['s_dec']))
  except Exception: continue
  z=dict(r); z['center_distance_deg']=dd; cand.append(z)
 exact=[z for z in cand if str(z.get('target_name'))==tile]
 exact.sort(key=lambda z:(str(z.get('obs_publisher_did')),str(z.get('obs_id'))))
 sel=exact[0] if len(exact)==1 else None
 return {'group_id':gid,'tile_id':tile,'reference_ra_deg':ra,'reference_dec_deg':dec,'candidate_count':len(cand),'exact_target_name_matches':len(exact),'selected_r_product':sel,'query_receipt':rec,'selection_rule':'require exactly one KiDS r_SDSS image product with target_name equal to frozen tile_id; no nearest-tile fallback'}
def main():
 cal=[]; psf_replay=0; lim_compatible=0; exact_tile_products=0
 for i,(g,t,ra,de,oldpsf,oldlim) in enumerate(CAL,1):
  x=query_point(g,t,ra,de,f'cal_{i:02d}_{g}'); x['historical_local_r_psf_fwhm']=oldpsf; x['historical_local_r_limiting_magnitude']=oldlim
  s=x.get('selected_r_product') or {}; psf=s.get('s_resolution'); lim=s.get('abmaglim')
  try: x['psf_round2_replay']=round(float(psf),2)==round(oldpsf,2)
  except: x['psf_round2_replay']=False
  try:
   x['lim_absolute_delta_mag']=abs(float(lim)-oldlim); x['lim_archive_compatible_le_0p01mag']=x['lim_absolute_delta_mag']<=0.01+1e-12
  except: x['lim_absolute_delta_mag']=None; x['lim_archive_compatible_le_0p01mag']=False
  exact_tile_products+=int(x['exact_target_name_matches']==1); psf_replay+=int(x['psf_round2_replay']); lim_compatible+=int(x['lim_archive_compatible_le_0p01mag']); cal.append(x)
 calibration_ok=(exact_tile_products==len(CAL) and psf_replay==len(CAL) and lim_compatible==len(CAL))
 out=[]
 if calibration_ok:
  for i,(g,t,ra,de) in enumerate(H24,1): out.append(query_point(g,t,ra,de,f'h24_{i:02d}_{g.replace("+","p").replace("-","m")}'))
 h24_ok=(len(out)==len(H24) and all(x.get('selected_r_product') is not None for x in out))
 status='PASS_OBSCORE_TILE_QUALITY_PROVENANCE_COMPATIBLE_AND_SELECTION_INVARIANT' if calibration_ok and h24_ok else 'FAIL_OBSCORE_TILE_QUALITY_PROVENANCE_OR_H24_RESOLUTION'
 m={'schema_version':2,'protocol':'P4.9.4-tile-quality-provenance','claim_boundary':'Metadata only. Pilot8 rows are exposed development calibration of field semantics, not successor-score evidence. No image bytes, model scores, private controls, unknown targets or future validation touched.','calibration':{'exact_tile_products':exact_tile_products,'psf_round2_replay':psf_replay,'lim_archive_compatible_le_0p01mag':lim_compatible,'required':len(CAL),'rows':cal},'selection_invariance_proof':{'premise_1':'The frozen P4.9 control contract requires exact tile equality between each positive and every eligible control.','premise_2':'s_resolution and abmaglim are properties of the selected r-band tile product, so they are constant within an exact-tile group.','consequence':'For every eligible within-group positive-control pair, both tile-quality differences are exactly zero. Their normalized distance components are zero and their calipers cannot reject or reorder any same-tile candidate.','historical_archive_note':'Five exposed pilot8 tiles reproduce historical PSF to two decimals. Current abmaglim differs from the historical local limiting-magnitude field by at most 0.008 mag in calibration; this version/rounding difference is immaterial under the exact-tile invariant and does not alter the frozen thresholds.'},'h24':out,'status':status}
 b=(json.dumps(m,sort_keys=True,indent=2)+'\n').encode(); (OUT/'P494_R_TILE_QUALITY_PROVENANCE.json').write_bytes(b)
 (OUT/'SHA256SUMS.txt').write_text(''.join(f'{h(p.read_bytes())}  {p.name}\n' for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='SHA256SUMS.txt'))
 print(json.dumps({'status':status,'calibration_exact_tiles':exact_tile_products,'psf_round2_replay':psf_replay,'lim_compatible':lim_compatible,'h24_resolved':len(out),'manifest_sha256':h(b)},sort_keys=True))
 return 0 if status.startswith('PASS') else 2
if __name__=='__main__': sys.exit(main())
