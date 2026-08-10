from __future__ import annotations
import hashlib, importlib.util, json, shutil, sys, tarfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from .signed_utility_router import SignedUtilityCalibration, fit_signed_utility_calibration

TRAINING_RELEASE="2026-07-01"
TARGET_RELEASE="2026-09-01"
TARGET_ARCHIVE_NAME="CATNAP_2026_09_01.tar.gz"
TARGET_REQUIRED_FILES=("assay_2026-09-01.txt","virseqs_aa_O_2026-09-01.fasta")
FOLD_COUNT=5
EXPECTED_FIT_COUNTS={"oof_raw":5,"oof_residual":5,"oof_router":5,
"full_raw":1,"full_residual":1,"full_router":1,
"utility_positive":1,"utility_negative":1}
EXPECTED_TOTAL_FITS=20
REGIMES=("resistant","middle","potent")
_EPS=1e-12

def sha256_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def canonical_json(x:Any)->str:
    return json.dumps(x,sort_keys=True,indent=2,ensure_ascii=False)+"\n"

@dataclass
class FitLedger:
    counts:dict[str,int]=field(default_factory=lambda:{k:0 for k in EXPECTED_FIT_COUNTS})
    events:list[dict[str,Any]]=field(default_factory=list)
    def record(self,category:str,fold:int|None=None)->None:
        if category not in EXPECTED_FIT_COUNTS: raise ValueError("unknown fit category")
        self.counts[category]+=1
        self.events.append({"ordinal":len(self.events)+1,"category":category,"fold":fold})
        if self.counts[category]>EXPECTED_FIT_COUNTS[category]:
            raise RuntimeError(f"fit budget exceeded for {category}")
    @property
    def total(self)->int:return sum(self.counts.values())
    def assert_exact(self)->None:
        if self.counts!=EXPECTED_FIT_COUNTS or self.total!=EXPECTED_TOTAL_FITS:
            raise RuntimeError(f"fit ledger mismatch: {self.counts}")

@dataclass(frozen=True)
class TargetCommitment:
    status:str;release:str;archive_name:str;archive_sha256:str;archive_bytes:int
    @classmethod
    def load(cls,path:Path)->"TargetCommitment":
        p=json.loads(path.read_text())
        x=cls(str(p.get("status","")),str(p.get("release","")),
              str(p.get("archive_name","")),str(p.get("archive_sha256","")),
              int(p.get("archive_bytes",-1)));x.validate();return x
    def validate(self)->None:
        if self.status!="TARGET_ARCHIVE_HASH_COMMITTED":raise ValueError("target archive commitment is not closed")
        if self.release!=TARGET_RELEASE:raise ValueError("target release drifted")
        if self.archive_name!=TARGET_ARCHIVE_NAME:raise ValueError("target archive name drifted")
        if len(self.archive_sha256)!=64:raise ValueError("target archive SHA-256 missing")
        int(self.archive_sha256,16)
        if self.archive_bytes<=0:raise ValueError("target archive byte count missing")

@dataclass(frozen=True)
class DesignSet:
    row_id:np.ndarray;antibody_id:np.ndarray;antibody_family:np.ndarray
    virus_id:np.ndarray;virus_cluster:np.ndarray;truth:np.ndarray|None
    x_raw:np.ndarray;x_residual:np.ndarray;x_router:np.ndarray
    antibody_embedding:np.ndarray;virus_embedding:np.ndarray
    def validate(self,require_truth:bool)->None:
        a=(self.row_id,self.antibody_id,self.antibody_family,self.virus_id,
           self.virus_cluster,self.x_raw,self.x_residual,self.x_router,
           self.antibody_embedding,self.virus_embedding);n=len(self.row_id)
        if n==0 or any(len(v)!=n for v in a):raise ValueError("design arrays must be nonempty and row-aligned")
        if require_truth and self.truth is None:raise ValueError("training truth is required")
        if self.truth is not None and len(self.truth)!=n:raise ValueError("truth is not row-aligned")
        if len(set(map(str,self.row_id)))!=n:raise ValueError("row IDs must be unique")
        nums=[self.x_raw,self.x_residual,self.x_router,self.antibody_embedding,self.virus_embedding]
        if self.truth is not None:nums.append(self.truth)
        if not all(np.all(np.isfinite(np.asarray(v,dtype=float))) for v in nums):
            raise ValueError("non-finite design input")
        if len(np.unique(self.virus_cluster.astype(str)))<5:raise ValueError("fewer than five virus clusters")

class SourceAdapter(Protocol):
    adapter_sha256:str
    def build_training_design(self,training_data:Path,families:Path)->DesignSet:...
    def build_target_design(self,target_data:Path,families:Path,training_design:DesignSet)->DesignSet:...

def assign_group_folds(virus_cluster:Sequence[str],row_id:Sequence[str],fold_count:int=5)->np.ndarray:
    c=np.asarray(virus_cluster,dtype=str);r=np.asarray(row_id,dtype=str)
    if len(c)==0 or len(c)!=len(r):raise ValueError("cluster and row arrays must be nonempty and aligned")
    if fold_count!=5:raise ValueError("the frozen contract requires exactly five folds")
    counts=Counter(c.tolist());ordered=sorted(counts,key=lambda x:(-counts[x],x))
    rows=[0]*5;groups=[0]*5;assignment={}
    for cluster in ordered:
        fold=min(range(5),key=lambda i:(rows[i],groups[i],i))
        assignment[cluster]=fold;rows[fold]+=counts[cluster];groups[fold]+=1
    out=np.asarray([assignment[x] for x in c],dtype=int)
    if set(out.tolist())!=set(range(5)):raise ValueError("deterministic assignment did not populate all five folds")
    if any(len(np.unique(out[c==x]))!=1 for x in ordered):raise AssertionError("a virus cluster leaked across folds")
    return out

def regime_labels(y:np.ndarray)->np.ndarray:
    y=np.asarray(y,float);return np.where(y<=-4,"resistant",np.where(y>=0,"potent","middle"))

@dataclass(frozen=True)
class OrthogonalProjection:
    coefficients:np.ndarray
    def transform(self,x:np.ndarray,raw:np.ndarray)->np.ndarray:
        b=np.column_stack([np.ones(len(raw)),raw]);return np.asarray(x,float)-b@self.coefficients

def fit_orthogonal_projection(x:np.ndarray,raw:np.ndarray)->tuple[np.ndarray,OrthogonalProjection]:
    x=np.asarray(x,float);raw=np.asarray(raw,float);b=np.column_stack([np.ones(len(raw)),raw])
    coef,*_=np.linalg.lstsq(b,x,rcond=None);z=x-b@coef
    mean=float(np.max(np.abs(np.mean(z,axis=0))))
    inner=float(np.max(np.abs(raw@z))/max(np.linalg.norm(raw)*np.linalg.norm(z),_EPS))
    if mean>1e-10 or inner>1e-10:raise RuntimeError("orthogonality gate failed")
    return z,OrthogonalProjection(coef)

def _entity_table(ids:np.ndarray,e:np.ndarray)->np.ndarray:
    d={}
    for k,v in zip(ids.astype(str),e,strict=True):
        if k in d and not np.array_equal(d[k],v):raise ValueError("entity embedding drift")
        d.setdefault(k,np.asarray(v,float))
    return np.vstack([d[k] for k in sorted(d)])

def _scale(ref:np.ndarray)->float:
    if len(ref)<2:raise ValueError("at least two reference entities are required")
    d=np.sqrt(np.sum((ref[:,None,:]-ref[None,:,:])**2,axis=2));np.fill_diagonal(d,np.inf)
    return max(float(np.median(d.min(axis=1))),1e-12)

def novelty_gate(train:DesignSet,query:DesignSet)->np.ndarray:
    ar=_entity_table(train.antibody_id,train.antibody_embedding)
    vr=_entity_table(train.virus_id,train.virus_embedding)
    ad=np.sqrt(np.sum((query.antibody_embedding[:,None,:]-ar[None,:,:])**2,axis=2)).min(axis=1)/_scale(ar)
    vd=np.sqrt(np.sum((query.virus_embedding[:,None,:]-vr[None,:,:])**2,axis=2)).min(axis=1)/_scale(vr)
    return 1/(1+.5*(np.maximum(ad-1,0)+np.maximum(vd-1,0)))

@dataclass(frozen=True)
class SuccessorResult:
    prediction:np.ndarray;raw_prediction:np.ndarray;residual_correction:np.ndarray
    p_resistant:np.ndarray;p_middle:np.ndarray;p_potent:np.ndarray
    fold_id:np.ndarray;calibration:SignedUtilityCalibration;fit_ledger:FitLedger

def _slice(d:DesignSet,m:np.ndarray)->DesignSet:
    return DesignSet(d.row_id[m],d.antibody_id[m],d.antibody_family[m],d.virus_id[m],
      d.virus_cluster[m],None if d.truth is None else d.truth[m],d.x_raw[m],
      d.x_residual[m],d.x_router[m],d.antibody_embedding[m],d.virus_embedding[m])

def _fit_base(train:DesignSet,valid:DesignSet,ledger:FitLedger,prefix:str,fold:int|None):
    if train.truth is None:raise ValueError("training truth missing")
    raw=Ridge(alpha=10,fit_intercept=True,solver="lsqr",tol=1e-8)
    ledger.record(prefix+"_raw",fold);raw.fit(train.x_raw,train.truth)
    raw_train=raw.predict(train.x_raw);raw_valid=raw.predict(valid.x_raw)
    xr,proj=fit_orthogonal_projection(train.x_residual,raw_train)
    residual=Ridge(alpha=10,fit_intercept=False,solver="lsqr",tol=1e-8)
    ledger.record(prefix+"_residual",fold);residual.fit(xr,train.truth-raw_train)
    correction=residual.predict(proj.transform(valid.x_residual,raw_valid))*novelty_gate(train,valid)
    router=LogisticRegression(C=.1,class_weight="balanced",solver="lbfgs",max_iter=1000,tol=1e-8,random_state=626)
    ledger.record(prefix+"_router",fold);router.fit(train.x_router,regime_labels(train.truth))
    p=router.predict_proba(valid.x_router);cols={x:i for i,x in enumerate(router.classes_)}
    if set(cols)!=set(REGIMES):raise RuntimeError("router did not fit all three regimes")
    return raw_valid,correction,p[:,[cols[x] for x in REGIMES]]

def execute_designs(training:DesignSet,target:DesignSet)->SuccessorResult:
    training.validate(True);target.validate(False)
    if set(training.row_id.astype(str))&set(target.row_id.astype(str)):raise ValueError("training-target row overlap")
    fold_id=assign_group_folds(training.virus_cluster,training.row_id);ledger=FitLedger();n=len(training.row_id)
    ro=np.full(n,np.nan);co=np.full(n,np.nan);po=np.full((n,3),np.nan)
    for fold in range(5):
        vm=fold_id==fold;tm=~vm
        if set(training.virus_cluster[tm])&set(training.virus_cluster[vm]):raise AssertionError("OOF virus-cluster leakage")
        ro[vm],co[vm],po[vm]=_fit_base(_slice(training,tm),_slice(training,vm),ledger,"oof",fold)
    if not np.all(np.isfinite(np.column_stack([ro,co,po]))):raise RuntimeError("OOF predictions are incomplete")
    cal=fit_signed_utility_calibration(raw_prediction_oof=ro,residual_correction_oof=co,
      p_resistant_oof=po[:,0],p_potent_oof=po[:,2],truth_training=training.truth,
      actual_regime_training=regime_labels(training.truth),oof_fold_id=fold_id)
    ledger.record("utility_positive");ledger.record("utility_negative")
    raw,correction,p=_fit_base(training,target,ledger,"full",None)
    prediction=cal.predict(raw,correction,p[:,0],p[:,2]);ledger.assert_exact()
    return SuccessorResult(prediction,raw,correction,p[:,0],p[:,1],p[:,2],fold_id,cal,ledger)

def load_source_adapter(path:Path,expected_sha256:str)->SourceAdapter:
    if len(expected_sha256)!=64:raise ValueError("source adapter commitment hash missing")
    int(expected_sha256,16)
    if sha256_file(path)!=expected_sha256:raise ValueError("source adapter hash mismatch")
    spec=importlib.util.spec_from_file_location("lazarus_september_source_adapter",path)
    if spec is None or spec.loader is None:raise RuntimeError("cannot import source adapter")
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    adapter=getattr(mod,"ADAPTER",None)
    if adapter is None:
        factory=getattr(mod,"build_adapter",None)
        if not callable(factory):raise RuntimeError("source adapter must export ADAPTER or build_adapter()")
        adapter=factory()
    if getattr(adapter,"adapter_sha256",None)!=expected_sha256:raise ValueError("source adapter self-declared hash drifted")
    return adapter

def _rmse(y,p):return float(np.sqrt(np.mean((y-p)**2)))

def _bootstrap(y,candidate,reference,groups,mask,seed,reps):
    labels=np.asarray(groups,str)[mask];y=y[mask];candidate=candidate[mask];reference=reference[mask]
    u=np.asarray(sorted(set(labels)));idx={g:np.flatnonzero(labels==g) for g in u}
    if len(u)<2:raise ValueError("cluster bootstrap requires at least two groups")
    rng=np.random.default_rng(seed);v=np.empty(reps)
    for i in range(reps):
        chosen=np.concatenate([idx[g] for g in rng.choice(u,size=len(u),replace=True)])
        v[i]=np.mean((y[chosen]-reference[chosen])**2)-np.mean((y[chosen]-candidate[chosen])**2)
    return float(np.quantile(v,.05))

def evaluate_successor(target:DesignSet,result:SuccessorResult,bootstrap_replicates:int=5000)->dict[str,Any]:
    if target.truth is None:raise ValueError("target truth is required for scientific adjudication")
    if bootstrap_replicates!=5000:raise ValueError("the frozen contract requires exactly 5000 bootstrap replicates")
    y=np.asarray(target.truth,float);raw=result.raw_prediction;original=raw+result.residual_correction;s=result.prediction
    labels=regime_labels(y);point={}
    for scope in ("overall",*REGIMES):
        m=np.ones(len(y),bool) if scope=="overall" else labels==scope
        point[scope]={"entries":int(m.sum()),"raw_rmse":_rmse(y[m],raw[m]),
          "original_residual_rmse":_rmse(y[m],original[m]),"successor_rmse":_rmse(y[m],s[m])}
    boot={}
    for unit,groups,seed in (("antibody_family",target.antibody_family,26090101),("virus",target.virus_id,26090102)):
        boot[unit]={}
        for ri,(refname,ref) in enumerate((("raw",raw),("original_residual",original))):
            boot[unit][refname]={}
            for si,(scope,m) in enumerate((("overall",np.ones(len(y),bool)),("potent",labels=="potent"))):
                boot[unit][refname][scope]={"successor_mse_gain_lower_95":
                  _bootstrap(y,s,ref,groups,m,seed+10*ri+si,bootstrap_replicates),
                  "replicates":bootstrap_replicates}
    gp=all(point[x]["successor_rmse"]<point[x]["raw_rmse"] and
           point[x]["successor_rmse"]<point[x]["original_residual_rmse"]
           for x in ("overall","potent","middle","resistant"))
    gu=all(boot[u][r][s]["successor_mse_gain_lower_95"]>0 for u in boot
           for r in ("raw","original_residual") for s in ("overall","potent"))
    joint=(point["overall"]["successor_rmse"]<point["overall"]["raw_rmse"] and
      point["potent"]["successor_rmse"]<point["potent"]["raw_rmse"] and
      point["middle"]["successor_rmse"]<point["middle"]["original_residual_rmse"] and
      point["resistant"]["successor_rmse"]<point["resistant"]["original_residual_rmse"])
    status="UTILITY_ROUTER_GLOBAL_DOMINANCE_STRONG_PASS" if gp and gu else (
      "UTILITY_ROUTER_JOINT_REPLICATION_PASS" if joint else "UTILITY_ROUTER_FAILURE")
    return {"schema_version":"1.0","status":status,"point_metrics":point,"bootstrap":boot,
      "global_dominance_point_gates":gp,"global_dominance_uncertainty_gates":gu,
      "joint_replication_gates":joint,"bootstrap_replicates_per_unit":bootstrap_replicates}

def extract_committed_archive(archive:Path,commitment:TargetCommitment,destination:Path):
    if archive.name!=TARGET_ARCHIVE_NAME:raise ValueError("target archive filename drifted")
    if archive.stat().st_size!=commitment.archive_bytes:raise ValueError("target archive byte count mismatch")
    if sha256_file(archive)!=commitment.archive_sha256:raise ValueError("target archive hash mismatch")
    if destination.exists():raise FileExistsError(destination)
    destination.mkdir();manifest={}
    with tarfile.open(archive,"r:gz") as h:
        members={}
        for m in h.getmembers():
            if m.isfile():
                name=Path(m.name).name
                if name in members:raise ValueError("duplicate target member basename")
                members[name]=m
        if set(TARGET_REQUIRED_FILES)-set(members):raise ValueError("target archive required member missing")
        for name in TARGET_REQUIRED_FILES:
            src=h.extractfile(members[name])
            if src is None:raise ValueError("unreadable target member")
            out=destination/name
            with src,out.open("wb") as sink:shutil.copyfileobj(src,sink)
            manifest[name]={"bytes":out.stat().st_size,"sha256":sha256_file(out)}
    return manifest

def zero_target_preflight(commitment_path:Path|None,target_archive:Path|None,output:Path):
    if output.exists():raise FileExistsError("duplicate output path")
    if target_archive is not None:raise ValueError("zero-target preflight forbids a target archive path")
    status="ABSENT"
    if commitment_path is not None:
        p=json.loads(commitment_path.read_text());status=str(p.get("status",""))
        if status=="TARGET_ARCHIVE_HASH_COMMITTED":TargetCommitment.load(commitment_path)
    staging=output.parent/f".{output.name}.staging"
    if staging.exists():raise FileExistsError("stale staging path")
    staging.mkdir()
    receipt={"schema_version":"1.0","status":"ZERO_TARGET_PREFLIGHT_PASS",
      "training_release":TRAINING_RELEASE,"target_release":TARGET_RELEASE,
      "target_archive_name":TARGET_ARCHIVE_NAME,"target_commitment_status":status,
      "target_archive_path_received":False,"target_archive_bytes_read":0,
      "target_members_listed":0,"target_members_extracted":0,"target_rows_read":0,
      "target_values_read":0,"scientific_supervised_fits":0,
      "fit_budget_expected":EXPECTED_FIT_COUNTS,"fit_budget_total_expected":20,
      "fold_count":5,"duplicate_run_rejected":True,"atomic_publication":True}
    (staging/"SEPTEMBER_ZERO_TARGET_PREFLIGHT_RECEIPT.json").write_text(canonical_json(receipt))
    staging.rename(output);return receipt

def publish_scientific_outputs(output:Path,result:SuccessorResult,target:DesignSet,
 target_file_manifest:Mapping[str,Mapping[str,Any]],commitment:TargetCommitment,
 adapter_sha256:str,adjudication:Mapping[str,Any]|None=None):
    if output.exists():raise FileExistsError("duplicate sealed output path")
    staging=output.parent/f".{output.name}.staging"
    if staging.exists():raise FileExistsError("stale staging path")
    staging.mkdir()
    try:
        data=np.column_stack([target.row_id.astype(str),result.raw_prediction,
          result.residual_correction,result.p_resistant,result.p_middle,result.p_potent,result.prediction])
        np.savetxt(staging/"september_signed_utility_predictions.csv",data,delimiter=",",fmt="%s",
          header="row_id,raw_prediction,residual_correction,p_resistant,p_middle,p_potent,signed_utility_prediction",comments="")
        model={"schema_version":"1.0","family":"signed_utility_calibrated_soft_router_v1",
          "training_release":TRAINING_RELEASE,"target_release":TARGET_RELEASE,
          "target_archive_sha256":commitment.archive_sha256,"target_archive_bytes":commitment.archive_bytes,
          "target_files":dict(target_file_manifest),"source_adapter_sha256":adapter_sha256,
          "candidate_count":1,"adaptive_candidates":0,"fit_counts":result.fit_ledger.counts,
          "fit_events":result.fit_ledger.events,"total_supervised_fits":result.fit_ledger.total,
          "fold_count":5,"calibration":result.calibration.to_dict(),"target_batch_statistics_used":0,
          "atomic_publication":True}
        (staging/"SEPTEMBER_MODEL_RECEIPT.json").write_text(canonical_json(model))
        if adjudication is not None:(staging/"SEPTEMBER_ADJUDICATION.json").write_text(canonical_json(dict(adjudication)))
        files={p.name:{"bytes":p.stat().st_size,"sha256":sha256_file(p)} for p in staging.iterdir() if p.is_file()}
        receipt={"schema_version":"1.0","status":"SCIENTIFIC_OUTPUT_SEALED","files":files,
          "total_supervised_fits":result.fit_ledger.total,"candidate_count":1,
          "adaptive_candidates":0,"atomic_publication":True}
        (staging/"SEPTEMBER_RELAY_RECEIPT.json").write_text(canonical_json(receipt))
        staging.rename(output);return receipt
    except Exception:
        shutil.rmtree(staging,ignore_errors=True);raise
