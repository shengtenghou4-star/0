from __future__ import annotations
import csv, hashlib, json, pathlib, time
import requests

ENDPOINT='https://archive.eso.org/tap_obs/sync'
OUT=pathlib.Path('p495_h24_product_artifact')
TOP=20
TILES=['KIDS_137.0_0.5','KIDS_150.1_2.2','KIDS_159.0_0.5','KIDS_161.0_0.5','KIDS_188.5_-2.5','KIDS_204.0_1.5']
FIELDS=['obs_publisher_did','obs_id','target_name','obs_collection','instrument_name','facility_name','dataproduct_type','access_url','access_format','calib_level','s_ra','s_dec','em_min','em_max','t_min','t_max','filter']
OUT_FIELDS=['tile_id','band','obs_publisher_did','obs_id','filter','t_min','t_max','s_ra','s_dec','access_url','access_format','obs_collection','instrument_name','facility_name']

def sha(b): return hashlib.sha256(b).hexdigest()
def cbytes(v): return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def query(tile):
 return f'''SELECT TOP {TOP}\n  obs_publisher_did,\n  obs_id,\n  target_name,\n  obs_collection,\n  instrument_name,\n  facility_name,\n  dataproduct_type,\n  access_url,\n  access_format,\n  calib_level,\n  s_ra,\n  s_dec,\n  em_min,\n  em_max,\n  t_min,\n  t_max,\n  filter\nFROM ivoa.ObsCore\nWHERE target_name='{tile}'\n  AND dataproduct_type='image'\n  AND filter IN ('u_SDSS','g_SDSS','r_SDSS','i_SDSS')\nORDER BY filter, t_min, obs_publisher_did\n'''
def get(session,q):
 last=None
 for attempt in range(1,4):
  t=time.monotonic()
  try:
   r=session.post(ENDPOINT,data={'REQUEST':'doQuery','LANG':'ADQL','FORMAT':'json','QUERY':q},timeout=180)
   raw=r.content; rec={'attempt':attempt,'http_status':r.status_code,'content_type':r.headers.get('content-type'),'elapsed_seconds':time.monotonic()-t,'response_bytes':len(raw),'response_sha256':sha(raw)}
   r.raise_for_status(); return raw,rec
  except Exception as e:
   last=f'{type(e).__name__}: {e}'
   if attempt<3: time.sleep(2**(attempt-1))
 raise RuntimeError(last)
def parse(raw,tile):
 p=json.loads(raw); names=[str(x.get('name','')) for x in p.get('metadata',[])]
 if names!=FIELDS: raise RuntimeError(f'{tile}: response columns drift')
 rows=[dict(zip(names,r)) for r in p.get('data',[])]
 if not rows or len(rows)>=TOP: raise RuntimeError(f'{tile}: zero rows or TOP truncation')
 for r in rows:
  if r['target_name']!=tile or r['dataproduct_type']!='image': raise RuntimeError(f'{tile}: scope drift')
  if r['filter'] not in {'u_SDSS','g_SDSS','r_SDSS','i_SDSS'}: raise RuntimeError(f'{tile}: filter drift')
  if not str(r['obs_publisher_did']).startswith('ivo://eso.org/ID?ADP.'): raise RuntimeError(f'{tile}: bad DID')
 return rows
def select(tile,rows):
 by={f:[r for r in rows if r['filter']==f] for f in ['u_SDSS','g_SDSS','r_SDSS','i_SDSS']}
 for f in ['u_SDSS','g_SDSS','r_SDSS']:
  if len(by[f])!=1: raise RuntimeError(f'{tile}: {f} count {len(by[f])}')
 if len(by['i_SDSS'])!=2: raise RuntimeError(f"{tile}: i_SDSS count {len(by['i_SDSS'])}")
 ir=sorted(by['i_SDSS'],key=lambda r:(float(r['t_min']),float(r['t_max']),str(r['obs_id']),str(r['obs_publisher_did'])))
 chosen=[('u',by['u_SDSS'][0]),('g',by['g_SDSS'][0]),('r',by['r_SDSS'][0]),('i1',ir[0]),('i2',ir[1])]
 return [{k:v for k,v in [('tile_id',tile),('band',b),('obs_publisher_did',r['obs_publisher_did']),('obs_id',r['obs_id']),('filter',r['filter']),('t_min',r['t_min']),('t_max',r['t_max']),('s_ra',r['s_ra']),('s_dec',r['s_dec']),('access_url',r['access_url']),('access_format',r['access_format']),('obs_collection',r['obs_collection']),('instrument_name',r['instrument_name']),('facility_name',r['facility_name'])]} for b,r in chosen]
def main():
 OUT.mkdir(exist_ok=True); rawdir=OUT/'raw_json'; rawdir.mkdir(exist_ok=True)
 s=requests.Session(); s.headers['User-Agent']='HOU-LENS-P4.9.5-H24-product-metadata/1.0'
 products=[]; receipts=[]; plan=[]
 for i,tile in enumerate(TILES,1):
  q=query(tile); qsha=sha(q.encode()); plan.append({'order':i,'tile_id':tile,'top_limit':TOP,'query':q,'query_sha256':qsha})
  raw,tr=get(s,q); (rawdir/f'{i:03d}.json').write_bytes(raw); rows=parse(raw,tile); sel=select(tile,rows); products+=sel
  receipts.append({'order':i,'tile_id':tile,'query_sha256':qsha,'raw_response_bytes':len(raw),'raw_response_sha256':sha(raw),'resolved_product_count':5,'transport':tr})
  print(f'{i}/6 {tile}: {len(rows)} rows -> 5 products',flush=True)
 if len(products)!=30 or len({str(x['obs_publisher_did']) for x in products})!=30: raise RuntimeError('30-product uniqueness gate failed')
 plan_obj={'schema_version':1,'protocol':'HOU-LENS-P4.9.5-H24-PRODUCT-METADATA-ONLY','endpoint':'https://archive.eso.org/tap_obs','tiles':TILES,'queries':plan,'images_downloaded':False,'scores_computed':False}
 (OUT/'P495_H24_TILE_PRODUCT_QUERY_PLAN.json').write_bytes(cbytes(plan_obj))
 with (OUT/'P495_H24_TILE_PRODUCT_MANIFEST.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=OUT_FIELDS,lineterminator='\n'); w.writeheader(); w.writerows(products)
 raw_hashes=[{'order':i,'sha256':sha((rawdir/f'{i:03d}.json').read_bytes()),'bytes':len((rawdir/f'{i:03d}.json').read_bytes())} for i in range(1,7)]
 receipt={'schema_version':1,'protocol':'HOU-LENS-P4.9.5-H24-PRODUCT-METADATA-ONLY','status':'PASS_EXACT_6_TILE_30_PRODUCT_METADATA_CLOSURE','claim_boundary':'Metadata-only product resolution for six already-public H24 tiles. No SODA request, FITS byte, pixel, model feature, score, unknown target, or blind search was accessed.','tile_count':6,'selected_product_count':30,'bands_per_tile':['u','g','r','i1','i2'],'queries':receipts,'output_sha256':{'query_plan':sha((OUT/'P495_H24_TILE_PRODUCT_QUERY_PLAN.json').read_bytes()),'tile_product_manifest':sha((OUT/'P495_H24_TILE_PRODUCT_MANIFEST.csv').read_bytes()),'dataset_id_set':sha(cbytes(sorted(str(x['obs_publisher_did']) for x in products))),'raw_response_set':sha(cbytes(raw_hashes))},'images_downloaded':False,'image_bytes_downloaded':0,'scores_computed':False,'blind_search_performed':False,'unknown_dr5_only_targets_queried':False}
 (OUT/'P495_H24_TILE_PRODUCT_METADATA_RECEIPT.json').write_bytes(cbytes(receipt)); print(json.dumps(receipt,indent=2),flush=True)
if __name__=='__main__': main()
