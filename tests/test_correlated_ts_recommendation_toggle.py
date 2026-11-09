"""LCB switching must affect recommendations without changing TS sampling."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))


@pytest.fixture
def cli(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('cts_recommendation_cli_test',ROOT/'scripts/run_correlated_ts_baseline.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    matrix=tmp_path/'scores.npy';archive=tmp_path/'inputs.npz'
    np.save(matrix,np.array([[.45]*10,[.2]*10],dtype=np.float32))
    np.savez(archive,X=np.array([[0.],[1.]]),arm_ids=np.arange(2),cost=np.ones(2)*10)
    monkeypatch.chdir(tmp_path)
    def invoke(penalty,flag='--recommendation-std-penalty',output=None):
        args=['runner','--matrix',str(matrix),'--bo-inputs',str(archive),
              '--batch-size','4','--kernel-mode','fixed','--seed','0','--max-steps','1',
              '--eval-budget-fraction','1',flag,str(penalty)]
        if output is not None:args+=['--out',str(output)]
        monkeypatch.setattr(sys,'argv',args)
        return module.main()
    return module,invoke,tmp_path


def test_correlated_ts_toggle_changes_only_recommendation_and_keeps_both_outputs(cli):
    _,invoke,tmp=cli
    assert invoke(0,'--recommendation_std_penalty')==0
    assert invoke(1)==0
    paths=list((tmp/'outputs/correlated_ts').glob('*.npz'));assert len(paths)==2
    saved={}
    for p in paths:
        with np.load(p,allow_pickle=False) as z:
            metadata=json.loads(z['metadata_json'].item())
            beta=metadata['recommendation_std_penalty']
            saved[beta]={k:z[k].copy() for k in z.files}
            expected='finite_population_posterior_mean_minus_std' if beta else 'finite_population_posterior_mean'
            assert metadata['recommendation_rule']==expected
    for key in ['pulled_arm','sampled_columns','x','x_original_cost','n_new_samples','counts','observed_sums']:
        np.testing.assert_array_equal(saved[0][key],saved[1][key])
    assert saved[0]['recommended_arm'].tolist()==[1]
    assert saved[1]['recommended_arm'].tolist()==[0]


def test_preexisting_result_identity_stays_reusable_when_derived_rule_is_added(cli,monkeypatch):
    module,invoke,tmp=cli;output=tmp/'existing.npz'
    assert invoke(1,output=output)==0
    with np.load(output,allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    metadata=json.loads(data['metadata_json'].item())
    metadata.pop('recommendation_rule')
    data['metadata_json']=json.dumps(metadata)
    np.savez_compressed(output,**data)
    original=output.read_bytes()
    def forbidden(*args,**kwargs):
        raise AssertionError('A compatible completed experiment must not be run again')
    monkeypatch.setattr(module,'simulate_correlated_ts',forbidden)
    assert invoke(1,output=output)==0
    assert output.read_bytes()==original


@pytest.mark.parametrize('penalty',['-1','nan','inf'])
def test_invalid_penalty_is_rejected_before_loading_inputs(tmp_path,monkeypatch,penalty):
    spec=importlib.util.spec_from_file_location('cts_invalid_penalty_test',ROOT/'scripts/run_correlated_ts_baseline.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(sys,'argv',['runner','--matrix',str(tmp_path/'absent.npy'),
                                  '--recommendation-std-penalty',penalty])
    with pytest.raises(SystemExit) as error:module.main()
    assert error.value.code==2
