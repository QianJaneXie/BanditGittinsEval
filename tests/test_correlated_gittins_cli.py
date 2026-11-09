"""Verify the standalone entry point shares an offline cache and ignores Y."""
import importlib.util
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'src'))


def test_prepare_then_cached_cli_run_needs_no_online_precomputation(tmp_path,monkeypatch):
    import marginal_gittins_lookup
    matrix=np.array([[0,1,0,0,1,0,0,0,0,0,0],
                     [1,1,1,0,1,1,1,0,1,1,1],
                     [0,0,0,0,0,0,0,0,0,0,0]],dtype=np.float32)
    matrix_path=tmp_path/'scores.npy';archive=tmp_path/'inputs.npz';cache=tmp_path/'roots.npz'
    np.save(matrix_path,matrix)
    np.savez(archive,X=np.array([[0,0],[0,1],[1,0]]),arm_ids=np.arange(3),
             cost=np.ones(3)*11,Y=np.full((3,1),np.nan),raw_scores_mean=np.full(3,np.nan))
    spec=importlib.util.spec_from_file_location('correlated_gittins_cli',ROOT/'scripts/run_correlated_gittins_baseline.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    args=['runner','--matrix',str(matrix_path),'--bo-inputs',str(archive),
          '--root-cache',str(cache),'--batch-size','4','--eval-budget-fraction','1',
          '--kernel-mode','fixed','--variance-grid-size','5','--gittins-grid-points','257']
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys,'argv',args+['--prepare-only'])
    assert module.main()==0 and cache.is_file()
    assert not list((tmp_path/'outputs/correlated_gittins').glob('*.npz'))
    def forbidden(**kwargs):
        raise AssertionError('An existing root cache must not be recomputed during allocation')
    monkeypatch.setattr(marginal_gittins_lookup,'build_marginal_gittins_root_table',forbidden)
    monkeypatch.setattr(sys,'argv',args)
    assert module.main()==0
    files=list((tmp_path/'outputs/correlated_gittins').glob('*.npz'));assert len(files)==1
    with np.load(files[0],allow_pickle=False) as trace:
        assert trace['x'][-1]==matrix.size
        assert trace['regret'][-1]==0
        assert trace['independent_arm_optimality_guarantee'].item() is False
