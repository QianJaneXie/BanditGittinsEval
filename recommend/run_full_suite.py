"""Run the fixed primary Gittins+LCB comparison on AlpacaEval and all MMLU.

Existing smaller experiments are preserved. The compact representation retains
the complete observation stream and both recommendations after every update.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from full_suite_engine import run_path, METHOD_KEYS
from portable_gittins import compute_roots
from methods import method_specs

ROOT=Path(__file__).resolve().parents[1]
MMLU_PRIORS={"low":(.4,.02),"medium":(.6,.02),"high":(.75,.01)}
ALPACA=ROOT/"data/BanditEval_matrices/alpaca_eval_weighted_alpaca_eval_gpt4_turbo_2d_comparisons_no_rounding_debias.npy"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def specs(args):
    if args.benchmark in ["all","alpaca_eval"]:
        yield "alpaca_eval","AlpacaEval",ALPACA,(.2,.01),16
    if args.benchmark in ["all","mmlu"]:
        metadata=json.loads((ROOT/"data/MMLU_matrices/task_metadata.json").read_text())
        entries={r["task"]:r["dataset_prior_bucket"] for r in metadata["tasks"]}
        if set(entries)!={p.stem for p in (ROOT/"data/MMLU_matrices").glob("*.npy")}:
            raise ValueError("MMLU metadata and available subject matrices disagree")
        for task in (args.tasks or sorted(entries)):
            yield "mmlu_"+task,"MMLU: "+task.replace("_"," ").title(),ROOT/f"data/MMLU_matrices/{task}.npy",MMLU_PRIORS[entries[task]],4


def run_subject(job):
    slug,label,path,dataset_prior,batch_size,out_root,run_seeds,resume=job
    out=Path(out_root)/slug
    (out/"traces").mkdir(parents=True,exist_ok=True)
    (out/"roots").mkdir(exist_ok=True)
    path=Path(path)
    raw=np.load(path,allow_pickle=False)
    if raw.ndim!=2 or not np.isfinite(raw).all() or raw.min()<0 or raw.max()>1:
        raise ValueError(f"Expected complete bounded [0,1] score matrix: {path}")
    matrix=raw.astype(np.float64)
    priors={"default":(.5,.04),"dataset":dataset_prior}
    config=dict(benchmark=slug,benchmark_label=label,priors=list(priors),matrix_seeds=[1],
        run_seeds=run_seeds,batch_size=batch_size,budget_fraction=.1,cost_scale=1e-4,
        tau_sq_cell=.25,grid_points=1025,primary_method=METHOD_KEYS[1],
        dp_backend="NumPy/SciPy float64 m-diff FFT",sampling_generator="numpy.random.PCG64",
        input_arithmetic="float64 scores, batch sums, and oracle means",
        trace_format="compact-primary-v1")
    source_paths=[Path(__file__).resolve(),ROOT/"recommend/full_suite_engine.py",
        ROOT/"recommend/portable_gittins.py",ROOT/"recommend/methods.py",
        ROOT/"data/MMLU_matrices/task_metadata.json"]
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),status="running",config=config,
        priors=priors,methods=[asdict(s) for s in method_specs() if s.key in METHOD_KEYS],
        matrices={"1":dict(path=str(path.relative_to(ROOT)),sha256=digest(path),shape=list(matrix.shape),
            best_mean=float(matrix.mean(1).max()),binary=bool(np.isin(matrix,[0,1]).all()))},
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in source_paths},
        versions={p:importlib.metadata.version(p) for p in ["numpy","scipy"]},
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        design="Fixed Gaussian Gittins allocation; paired posterior mean vs fixed z=1.645 latent Gaussian LCB; exact round(10% matrix cells), no early stop.")
    target=out/"manifest.json"
    if target.exists():
        if not resume:
            raise FileExistsError(f"Use --resume for existing {out}")
        previous=json.loads(target.read_text())
        current=json.loads(json.dumps(manifest))
        for field in ["config","priors","methods","matrices","source_sha256","versions"]:
            if previous[field]!=current[field]:
                raise ValueError(f"Cannot resume changed {field}: {slug}")
        if previous["status"]=="complete" and len(list((out/"traces").glob("*.npz")))==2*len(run_seeds):
            return slug,previous["completed_runs"],previous["post_pull_states"],0.0
        manifest["created_utc"]=previous["created_utc"]
    target.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    started=time.perf_counter()
    total_steps=completed=0
    roots_hashes={}
    for prior,prior_pair in priors.items():
        roots=compute_roots(prior_pair[1],matrix.shape[1],config["cost_scale"])
        roots_path=out/"roots"/f"{prior}_N{matrix.shape[1]}.npy"
        np.save(roots_path,roots)
        roots_hashes[str(roots_path.relative_to(out))]=digest(roots_path)
        for seed in run_seeds:
            filename=out/"traces"/f"{prior}_matrix1_run{seed:02d}.npz"
            if resume and filename.exists():
                with np.load(filename) as saved:
                    assert saved["x"][-1]==round(.1*matrix.size)
                    total_steps+=len(saved["x"])
                completed+=1
                continue
            result=run_path(matrix,roots,prior_tag=prior,prior_pair=prior_pair,run_seed=seed,batch_size=batch_size)
            result["benchmark"]=np.asarray(slug)
            np.savez_compressed(filename,**result)
            completed+=1
            total_steps+=len(result["x"])
            if completed%10==0:
                print(f"{slug}: {completed}/{2*len(run_seeds)} runs; {time.perf_counter()-started:.1f}s",flush=True)
    manifest.update(status="complete",completed_runs=completed,post_pull_states=total_steps,
        roots_sha256=roots_hashes,elapsed_seconds=time.perf_counter()-started,
        completed_utc=datetime.now(timezone.utc).isoformat())
    target.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return slug,completed,total_steps,manifest["elapsed_seconds"]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark",choices=["all","alpaca_eval","mmlu"],default="all")
    parser.add_argument("--tasks",nargs="+")
    parser.add_argument("--run-seeds",nargs="+",type=int,default=list(range(20)))
    parser.add_argument("--workers",type=int,default=3)
    parser.add_argument("--out-root",type=Path,default=ROOT/"recommend/results/full_suite")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    if args.workers<1 or len(set(args.run_seeds))!=len(args.run_seeds) or any(s<0 for s in args.run_seeds):
        parser.error("Need positive workers and unique nonnegative sampling seeds")
    selected=list(specs(args))
    if len({s[0] for s in selected})!=len(selected):
        parser.error("Duplicate subjects")
    args.out_root.mkdir(parents=True,exist_ok=True)
    suite_path=args.out_root/"suite_manifest.json"
    suite=dict(status="running",datasets=[s[0] for s in selected],run_seeds=args.run_seeds,
        expected_runs=len(selected)*len(args.run_seeds)*2,
        interpretation="AlpacaEval plus all selected MMLU subjects evaluated separately; aggregate subject-level results, not a pooled-row recommendation.")
    suite_path.write_text(json.dumps(suite,indent=2),encoding="utf-8")
    # Long subjects start first so process workers finish at similar times.
    selected.sort(key=lambda s:np.load(s[2],mmap_mode="r").size,reverse=True)
    jobs=[(*s,str(args.out_root.resolve()),args.run_seeds,args.resume) for s in selected]
    started=time.perf_counter()
    totals=[]
    if args.workers==1:
        for job in jobs:
            totals.append(run_subject(job))
            print("COMPLETE",totals[-1],flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(run_subject,job) for job in jobs]
            for future in as_completed(futures):
                totals.append(future.result())
                print(f"COMPLETE {len(totals)}/{len(jobs)}: {totals[-1]}",flush=True)
    suite.update(status="complete",completed_runs=sum(t[1] for t in totals),post_pull_states=sum(t[2] for t in totals),
        elapsed_seconds=time.perf_counter()-started,completed_utc=datetime.now(timezone.utc).isoformat())
    assert suite["completed_runs"]==suite["expected_runs"]
    suite_path.write_text(json.dumps(suite,indent=2),encoding="utf-8")
    print(json.dumps(suite,indent=2),flush=True)


if __name__=="__main__":
    main()
