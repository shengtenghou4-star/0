#!/usr/bin/env python3
from __future__ import annotations
import hashlib, io, json, pathlib, sys, warnings
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

ROOT=pathlib.Path(__file__).resolve().parents[1]
IMAGING=ROOT/'hou_lens_p495_historical_south'/'source12_exact'
sys.path.insert(0,str(IMAGING))
import kids_imaging

INPUT=pathlib.Path('private_input')
OUTPUT=pathlib.Path('private_output')
MANIFEST_SHA='1528878b44099ceb27a7d65a364d57d0a018cd74b7b3eac005d48ce3750c7ae7'
RESULT_CERT_SHA='37571101784b05edfcbb7e697685e1d6c69356c2c44c86a5c3a3158f6cccee16'
EXPECTED_GROUPS={'KIDSOBJ-00819','KIDSOBJ-01340','KIDSOBJ-01427','KIDSOBJ-01951','KIDSOBJ-02104'}
EXPECTED_IMAGING_BLOB='583b5929c05f46d671228cbccc52af5709a2caa0'
BANDS=('u','g','r','i1','i2')

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def blob_sha(p):
 b=pathlib.Path(p).read_bytes(); return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()

def canonical(v): return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()

def inspect(path,row,band):
 body=path.read_bytes()
 with warnings.catch_warnings():
  warnings.simplefilter('ignore')
  with fits.open(io.BytesIO(body),memmap=False,checksum=True) as hdul:
   h=hdul[0]
   if h.data is None or h.data.ndim!=2: raise RuntimeError('2D FITS gate failed')
   image=np.asarray(h.data,dtype=np.float32)
   if image.shape!=(360,360): raise RuntimeError('360x360 geometry gate failed')
   w=WCS(h.header); x,y=w.world_to_pixel_values(float(row['ra_deg']),float(row['dec_deg']))
   if not (0<=x<360 and 0<=y<360): raise RuntimeError('target containment gate failed')
   if float(np.hypot(x-179.5,y-179.5))>2.0: raise RuntimeError('target centering gate failed')
   m=w.pixel_scale_matrix
   scales=[float(np.hypot(m[0,0],m[1,0])*3600),float(np.hypot(m[0,1],m[1,1])*3600)]
   if max(abs(v-0.2) for v in scales)>1e-5: raise RuntimeError('pixel scale gate failed')
   filt=str(h.header.get('FILTER',''))
   if not filt.lower().startswith(band[0]): raise RuntimeError('filter gate failed')
   psf=float(h.header.get('PSF_FWHM',np.nan))
   if not np.isfinite(psf) or psf<=0: raise RuntimeError('PSF gate failed')
   finite=float(np.isfinite(image).mean())
   if finite<=0.99: raise RuntimeError('finite coverage gate failed')
   return image,psf,float(np.median(scales)),{'shape':[360,360],'filter':filt,'psf_fwhm_arcsec':psf,'pixel_scale_arcsec':scales,'finite_fraction':finite,'target_pixel':[float(x),float(y)]}

def main():
 mp=INPUT/'five_positive_manifest.json'; cert=INPUT/'result-recipient-cert.pem'
 if sha(mp)!=MANIFEST_SHA: raise RuntimeError('private manifest SHA gate failed')
 if sha(cert)!=RESULT_CERT_SHA: raise RuntimeError('result certificate SHA gate failed')
 imaging_path=IMAGING/'kids_imaging.py'
 if blob_sha(imaging_path)!=EXPECTED_IMAGING_BLOB: raise RuntimeError('historical imaging core blob gate failed')
 m=json.loads(mp.read_text())
 if m.get('positive_count')!=5 or {x['group_id'] for x in m['positives']}!=EXPECTED_GROUPS: raise RuntimeError('five-positive identity gate failed')
 OUTPUT.mkdir(exist_ok=True)
 receipts=[]
 for idx,row in enumerate(sorted(m['positives'],key=lambda x:x['group_id']),1):
  od=OUTPUT/f'p{idx:02d}'; od.mkdir(exist_ok=True)
  by={b['band']:b for b in row['bands']}
  if set(by)!=set(BANDS): raise RuntimeError('five-band manifest gate failed')
  images={}; psf={}; pix=[]; inputs=[]
  for band in BANDS:
   meta=by[band]; p=INPUT/meta['file']
   if not p.exists() or p.stat().st_size!=meta['bytes'] or sha(p)!=meta['sha256']: raise RuntimeError('input FITS hash/byte gate failed')
   image,pf,px,fmeta=inspect(p,row,band); images[band]=image; psf[band]=pf; pix.append(px)
   inputs.append({'band':band,'dataset_id':meta['dataset_id'],'bytes':meta['bytes'],'sha256':meta['sha256'],'fits':fmeta})
  if max(pix)-min(pix)>1e-6: raise RuntimeError('pixel-scale consistency gate failed')
  px=float(np.median(pix))
  ugri,ugri_psf,i_meta=kids_imaging.prepare_ugri_from_kids_visits(images,psf,pixscale_arcsec=px)
  matched,target_psf,sigmas=kids_imaging.psf_match_bands(ugri,ugri_psf,pixscale_arcsec=px)
  raw=np.stack([matched[b] for b in ('u','g','r','i')],axis=0).astype(np.float32)
  if raw.shape!=(4,360,360) or float(np.isfinite(raw).mean())!=1.0: raise RuntimeError('raw cube gate failed')
  rawp=od/'psf_matched_ugri_scoring_cube.npz'
  np.savez_compressed(rawp,image=raw,bands=np.asarray(['u','g','r','i']),target_id=np.asarray(row['group_id']),tile_id=np.asarray(row['tile_id']),pixel_scale_arcsec=np.asarray(px),target_psf_fwhm_arcsec=np.asarray(target_psf))
  norm={}; norm_meta={}
  for b in ('u','g','r','i'): norm[b],norm_meta[b]=kids_imaging.robust_background_normalize(matched[b])
  model=np.stack([norm[b] for b in ('u','g','r','i')],axis=0).astype(np.float16)
  if model.shape!=(4,360,360) or float(np.isfinite(model).mean())!=1.0: raise RuntimeError('normalized cube gate failed')
  modelp=od/'normalized_ugri_model_cube.npz'
  np.savez_compressed(modelp,image=model,bands=np.asarray(['u','g','r','i']),target_id=np.asarray(row['group_id']),tile_id=np.asarray(row['tile_id']),pixel_scale_arcsec=np.asarray(px),target_psf_fwhm_arcsec=np.asarray(target_psf))
  rec={'schema_version':1,'protocol':'HOU-LENS-P4.9.5-POSITIVE-BASELINE-CUBE','private_index':f'p{idx:02d}','group_id':row['group_id'],'object_key':row['object_key'],'source_id':row['source_id'],'tile_id':row['tile_id'],'input_fits':inputs,'historical_imaging_core_git_blob_sha':EXPECTED_IMAGING_BLOB,'pixel_scale_arcsec':px,'i_visit_combination':i_meta.to_dict(),'ugri_input_psf_fwhm_arcsec':ugri_psf,'target_psf_fwhm_arcsec':target_psf,'convolution_sigma_pixels':sigmas,'normalization':norm_meta,'raw_cube_shape':list(raw.shape),'normalized_cube_shape':list(model.shape),'raw_cube_finite_fraction':float(np.isfinite(raw).mean()),'normalized_cube_finite_fraction':float(np.isfinite(model).mean()),'raw_cube_sha256':sha(rawp),'normalized_cube_sha256':sha(modelp),'scores_computed':False,'model_code_executed':False}
  (od/'receipt.json').write_bytes(canonical(rec)); receipts.append(rec)
  print(f'positive {idx}/5 cube-complete',flush=True)
 files=[]
 for p in sorted(x for x in OUTPUT.rglob('*') if x.is_file()): files.append({'path':p.relative_to(OUTPUT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
 aggregate={'schema_version':1,'protocol':'HOU-LENS-P4.9.5-FIVE-POSITIVE-BASELINE-CUBE-CONSTRUCTION','status':'PASS_5_OF_5_POSITIVE_BASELINE_CUBES','positive_count':5,'raw_cube_count':5,'normalized_cube_count':5,'historical_imaging_core_git_blob_sha':EXPECTED_IMAGING_BLOB,'input_manifest_sha256':MANIFEST_SHA,'output_file_set_before_aggregate':files,'output_file_set_sha256_before_aggregate':hashlib.sha256(canonical(files)).hexdigest(),'receipts':receipts,'scores_computed':False,'model_code_executed':False,'scenario_generation':False,'future_validation_touched':False}
 (OUTPUT/'aggregate_receipt.json').write_bytes(canonical(aggregate))
 print(json.dumps({'status':aggregate['status'],'positive_count':5,'file_set_sha256':aggregate['output_file_set_sha256_before_aggregate']},indent=2),flush=True)
if __name__=='__main__': main()
