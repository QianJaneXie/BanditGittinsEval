#!/usr/bin/env python3
"""Compare the new marginal-index heuristic with two cached baselines."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mplconfig'))
os.environ.setdefault('MPLBACKEND','Agg')
import matplotlib.pyplot as plt

from compare_correlated_gittins import MODEL_VERSION, POLICY, BASELINES
from compare_correlated_ts import _case_names, _validate_case_identity, _validate_reference_case
from plot_correlated_ts_comparison import paired_stats
from plot_gittins_ts_comparison import budget_average_regret, mean_and_se, previous_regret

METHODS=('gittins','correlated_ts',POLICY)
LABELS={'gittins':'Independent Gittins','correlated_ts':'Correlated Gaussian TS',POLICY:'Correlation-aware Gittins (marginal)'}
COLORS={'gittins':'#2563a6','correlated_ts':'#7c4bb0',POLICY:'#d36b20'}
PRIORS={'G':(.5,.04),'S':(.2,.01)}


def load_cases(directory):
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest.get('model_version')!=MODEL_VERSION or manifest.get('max_steps') is not None:
        raise ValueError('A complete marginal-index study is required for comparison')
    if manifest.get('optimality_guarantee_for_correlated_arms') is not False:
        raise ValueError('Missing marginal-index approximation metadata')
    cases=[]; matrices={}
    from compare_gittins_ts import load_paper_costs
    costs=load_paper_costs()
    for name,matrix,seed,prior,mode in _case_names(manifest):
        if matrix not in matrices:
            path=ROOT/f'data/BanditEval_matrices/gsm8k_1_samples_various_models_seed{matrix}.npy'
            if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['matrix_sha256'][str(matrix)]:
                raise ValueError('Response matrix changed after the experiment')
            matrices[matrix]=np.load(path,allow_pickle=False)
        truth=np.asarray(matrices[matrix],dtype=np.float64).mean(1)
        case=dict(matrix_seed=matrix,run_seed=seed,prior=prior,mode=mode,methods={})
        path=directory/'runs'/name
        with np.load(path,allow_pickle=False) as z:
            _validate_case_identity(z,manifest,(matrix,seed,prior,mode),path)
            for method in METHODS:
                if method==POLICY:
                    source=z; source_path=path; close=False
                else:
                    source_path=Path(z[f'{method}_reference_case'].item())
                    if hashlib.sha256(source_path.read_bytes()).hexdigest()!=z[f'{method}_reference_sha256'].item():
                        raise ValueError(f'Cached baseline changed: {source_path}')
                    source=np.load(source_path,allow_pickle=False); close=True
                    if source['settings_hash'].item()!=z[f'{method}_reference_settings_hash'].item():
                        raise ValueError('Cached baseline settings changed')
                    for key,expected in zip(('matrix_seed','run_seed','prior_name','cost_mode'),(matrix,seed,prior,mode)):
                        if source[key].item()!=expected:raise ValueError('Cached baseline case identity differs')
                try:
                    _validate_reference_case(source,manifest,(matrix,seed,prior,mode),matrices[matrix],costs,source_path,methods=(method,))
                    axis='x' if mode=='unit' else 'x_original_cost'
                    x=np.asarray(source[f'{method}_{axis}'],dtype=float)/float(source[f'{method}_full_budget'])
                    y=truth.max()-truth[np.asarray(source[f'{method}_recommended_arm'],dtype=int)]
                    initial=float(truth.max()-truth[0])
                    metrics={f'regret_at_{100*b:g}%':previous_regret(x,y,b,initial) for b in (.01,.02,.05,.1)}
                    metrics.update({f'auc_0_{100*b:g}%':budget_average_regret(x,y,b,initial) for b in (.02,.1)})
                    np.testing.assert_allclose(metrics['auc_0_10%'],source[f'{method}_auc_normalized'],atol=1e-12,rtol=0)
                    case['methods'][method]=dict(x=x,y=y,metrics=metrics,elapsed_s=float(source[f'{method}_elapsed_s']))
                finally:
                    if close:source.close()
        cases.append(case)
    return manifest,cases


def summarize(manifest,cases,directory):
    result=dict(protocol=manifest,approximation='Single-arm stopping roots ignore future cross-arm learning; no correlated-arm optimality claim',
                baselines_recomputed=False,uncertainty='Paired algorithm seeds within five response matrices; descriptive two-way bootstrap',panels={})
    rows=[]; differences=[]
    for mode in ('unit','cost'):
        for prior in PRIORS:
            panel=sorted([c for c in cases if c['mode']==mode and c['prior']==prior],key=lambda c:(c['matrix_seed'],c['run_seed']))
            item=dict(n_paired_trials=len(panel),methods={},paired_differences={})
            keys=list(panel[0]['methods'][POLICY]['metrics'])
            for method in METHODS:
                stats={}
                for metric in keys:
                    values=np.array([c['methods'][method]['metrics'][metric] for c in panel])
                    mean,se=mean_and_se(values)
                    stats[metric]=dict(mean=float(mean),se=float(se))
                    rows.append(dict(mode=mode,prior=prior,method=method,metric=metric,mean=float(mean),se=float(se)))
                stats['median_elapsed_s']=float(np.median([c['methods'][method]['elapsed_s'] for c in panel]))
                item['methods'][method]=stats
            for baseline in BASELINES:
                comparison={}
                for metric in keys:
                    delta=np.array([c['methods'][POLICY]['metrics'][metric]-c['methods'][baseline]['metrics'][metric] for c in panel])
                    stats=paired_stats(delta,panel)
                    stats['mean_difference']=stats.pop('mean_correlated_minus_independent')
                    stats['new_method_lower_fraction']=stats.pop('correlated_lower_fraction')
                    stats['baseline_lower_fraction']=stats.pop('independent_lower_fraction')
                    for per_matrix in stats['per_matrix']:
                        per_matrix['mean_difference']=per_matrix.pop('mean_correlated_minus_independent')
                    comparison[metric]=stats
                    differences.append(dict(mode=mode,prior=prior,baseline=baseline,metric=metric,**{k:v for k,v in stats.items() if k!='per_matrix'}))
                item['paired_differences'][f'{POLICY}_minus_{baseline}']=comparison
            result['panels'][f'{mode}_{prior}']=item
    for name,data in [('checkpoints.csv',rows),('paired_differences.csv',differences)]:
        with (directory/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in data for k in row)))
            writer.writeheader();writer.writerows(data)
    (directory/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def render(manifest,cases,directory,xmax,stem):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
    grid=np.linspace(0,xmax,350); curves={'grid_fraction':grid}; ymax=0.
    for row,mode in enumerate(('unit','cost')):
        for col,prior in enumerate(PRIORS):
            ax=axes[row,col]; selected=[c for c in cases if c['mode']==mode and c['prior']==prior]
            for method in METHODS:
                values=np.stack([np.interp(grid,c['methods'][method]['x'],c['methods'][method]['y'],left=np.nan,right=np.nan) for c in selected])
                mean,se=mean_and_se(values)
                ax.plot(100*grid,mean,color=COLORS[method],label=LABELS[method],lw=1.8)
                ax.fill_between(100*grid,np.maximum(0,mean-se),mean+se,color=COLORS[method],alpha=.16,lw=0)
                curves[f'{mode}_{prior}_{method}_mean']=mean; curves[f'{mode}_{prior}_{method}_se']=se
                if np.isfinite(mean).any():ymax=max(ymax,float(np.nanmax(mean+np.nan_to_num(se))))
            if row==0:
                mu,v=PRIORS[prior];ax.set_title(f"{'General' if prior=='G' else 'GSM8K'} prior: N({mu:g}, {v:g})",fontweight='semibold',pad=12)
            if col==0:ax.set_ylabel(f"{'Unit cost' if mode=='unit' else 'Heterogeneous cost'}\nSimple regret")
            ax.grid(axis='y',color='#dce4ed',lw=.7);ax.set_axisbelow(True);ax.set_xlim(0,100*xmax)
    for ax in axes.flat:ax.set_ylim(-.003,max(.02,1.06*ymax))
    fig.suptitle('GSM8K: correlation-aware Gittins with marginal indices',x=.085,y=.975,ha='left',fontsize=17,fontweight='semibold')
    fig.text(.085,.934,f"Joint categorical Gaussian beliefs; B={manifest['batch_size']}, LCB=1; {len(manifest['matrix_seeds'])} matrices x {len(manifest['run_seeds'])} paired trials",fontsize=10.5,color='#64748b')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.54,.905),ncol=3,frameon=False,fontsize=9.2)
    fig.supxlabel('Exhaustive-evaluation budget used (%)',y=.065,fontsize=11)
    fig.text(.085,.018,'Bands: +/-1 SE. Marginal-index heuristic ignores future cross-arm information sharing.',fontsize=8.8,color='#64748b')
    fig.subplots_adjust(left=.085,right=.985,top=.82,bottom=.13,hspace=.20,wspace=.15)
    for ext in ('png','pdf'):fig.savefig(directory/f'{stem}.{ext}',dpi=230,facecolor='white')
    plt.close(fig);np.savez_compressed(directory/f'{stem}_curves.npz',**curves)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir',type=Path,default=ROOT/'outputs/gsm8k_correlated_gittins_lcb100')
    a=p.parse_args();manifest,cases=load_cases(a.results_dir)
    result=summarize(manifest,cases,a.results_dir)
    render(manifest,cases,a.results_dir,.1,'comparison');render(manifest,cases,a.results_dir,.02,'comparison_early')
    for panel,item in result['panels'].items():
        print(panel,{m:round(item['methods'][m]['auc_0_10%']['mean'],8) for m in METHODS})


if __name__=='__main__':main()
