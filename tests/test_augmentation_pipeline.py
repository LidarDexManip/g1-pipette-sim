import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from g1_pipette_sim.augmentation.contracts import atomic_json, bind_run, digest, file_hash, lock, seal, verify
from g1_pipette_sim.augmentation.sources import load_episode
from g1_pipette_sim.augmentation.store import Ledger
from g1_pipette_sim.augmentation.splits import parent_splits


def test_cache_checks_contents_and_inputs(tmp_path):
    artifact=tmp_path/'result';artifact.write_text('first')
    seal(tmp_path,'screen',{'recipe':1},[artifact])
    assert verify(tmp_path,'screen',{'recipe':1})
    with pytest.raises(ValueError,match='Stale'):verify(tmp_path,'screen',{'recipe':2})
    artifact.write_text('other')
    with pytest.raises(ValueError,match='Corrupt'):verify(tmp_path,'screen',{'recipe':1})


def test_unversioned_run_cannot_be_adopted(tmp_path):
    (tmp_path/'screen.json').write_text('{}')
    with pytest.raises(ValueError,match='Unversioned'):bind_run(tmp_path,{'seed':1})


def test_changed_contract_fails_and_lock_excludes_other_coordinator(tmp_path):
    with lock(tmp_path):
        bind_run(tmp_path,{'seed':1})
        with pytest.raises(BlockingIOError):
            with lock(tmp_path):pass
    bind_run(tmp_path,{'seed':1})
    with pytest.raises(ValueError):bind_run(tmp_path,{'seed':2})


def test_all_command_channels_and_initial_state(monkeypatch,tmp_path):
    from g1_pipette_sim.augmentation import sources
    arrays={'action.low_level':np.zeros((3,29)), 'action.hand_left':np.zeros((3,6)),
            'action.hand_right':np.zeros((3,6)), 'observation.state':np.zeros((3,29))}
    ep=SimpleNamespace(arrays=arrays, action_times=np.arange(3.), names=['j']*29)
    monkeypatch.setattr(sources,'Episode',lambda root,index:ep)
    np.savez(tmp_path/'commands.npz',times=np.arange(3.),motor_targets=np.ones((3,29)),
             hand_left=np.full((3,6),123.),hand_right=np.full((3,6),456.),initial_state=np.zeros(29))
    (tmp_path/'config.json').write_text('{}')
    candidate={'parent_episode':14,'dataset_snapshot':'pinned', 'config':str(tmp_path/'config.json'), 'commands':str(tmp_path/'commands.npz')}
    candidate.update({key+'_sha256':file_hash(candidate[key]) for key in ['commands','config']})
    result=load_episode(candidate)
    assert np.all(result.arrays['action.low_level']==1)
    assert np.all(result.arrays['action.hand_left']==123)
    assert np.all(result.arrays['action.hand_right']==456)
    ep.arrays['observation.state'][0,0]=1
    with pytest.raises(ValueError,match='Initial'):load_episode(candidate)
    candidate['config_sha256']='bad'
    with pytest.raises(ValueError,match='config'):load_episode(candidate)


def test_ledger_recovers_interrupted_task_without_double_claim(tmp_path):
    ledger=Ledger(tmp_path);ledger.add('a');assert ledger.claim('a',2)
    assert not ledger.claim('a',2);ledger.close()
    ledger=Ledger(tmp_path);assert ledger.claim('a',2)
    ledger.finish('a','accepted',{'capture_id':'id'})
    assert not ledger.claim('a',2);ledger.close()
    ledger=Ledger(tmp_path)
    assert ledger.rows()[0]['state']=='accepted';ledger.close()


def test_retry_budget_is_bounded(tmp_path):
    ledger=Ledger(tmp_path);ledger.add('a')
    for _ in range(2):
        assert ledger.claim('a',2);ledger.finish('a','error',{})
    assert not ledger.claim('a',2);ledger.close()


def test_parent_splits_stable_when_catalog_grows(tmp_path):
    path=tmp_path/'split.json'
    first=parent_splits(path,tmp_path/'snapshot',194,7)
    second=parent_splits(path,tmp_path/'snapshot',200,7)
    assert all(first['parents'][key]==second['parents'][key] for key in first['parents'])
    with pytest.raises(ValueError):parent_splits(path,tmp_path/'snapshot',200,8)


def test_record_resumes_after_verification_without_acceptance(monkeypatch,tmp_path):
    from g1_pipette_sim.augmentation import runner
    c={'folder':str(tmp_path),'run_id':'r'}
    (tmp_path/'verification.json').write_text(json.dumps({'result':{'pickup':True}}))
    called=[]
    monkeypatch.setattr(runner,'load_episode',lambda c:None)
    def simulate(c,record):
        called.append(record)
        atomic_json(tmp_path/'verification.json',{'result':{'pickup':False}})
        return {'pickup':False}
    monkeypatch.setattr(runner,'validate',simulate)
    assert not runner.stage(c,True)['pickup'];assert called==[True]
    assert not runner.stage(c,True)['pickup'];assert called==[True]


def test_corrupt_acceptance_is_not_regenerated(monkeypatch,tmp_path):
    from g1_pipette_sim.augmentation import runner
    c={'folder':str(tmp_path),'run_id':'r'}
    monkeypatch.setattr(runner,'load_episode',lambda c:None)
    (tmp_path/'trajectory.npz').write_bytes(b'corrupt')
    atomic_json(tmp_path/'accepted.json',{'candidate':c,'artifact_hashes':{'trajectory.npz':'incorrect'}})
    monkeypatch.setattr(runner,'validate',lambda *a,**kw:pytest.fail('Must not overwrite corrupt acceptance'))
    with pytest.raises(ValueError,match='Corrupt'):runner.stage(c,True)


def test_motor_training_writer_crosses_chunk_boundary(tmp_path):
    from tools.prepare_starvla_motor_mix import job, metadata
    source=tmp_path/'source';source.mkdir()
    n=3
    np.savez(source/'trajectory.npz',timestamp=np.arange(n)/30,simulation_time=np.arange(n)/30,
             **{'observation.state':np.zeros((n,29)),'action':np.zeros((n,29)),
                'action.hand_left':np.ones((n,6))*123,'action.hand_right':np.ones((n,6))*456})
    for name in ['rgb','wrist_left','wrist_right']:(source/(name+'.mp4')).write_bytes(b'fixture')
    output=tmp_path/'out'
    rows=[job({'domain':'sim','parent_episode':14,'split':'train','episode_index':i,'output':str(output),'source':str(source)}) for i in [999,1000]]
    for i in [999,1000]:
        assert (output/f'data/chunk-{i//1000:03}/episode_{i:06}.parquet').exists()
        assert (output/f'videos/chunk-{i//1000:03}/observation.images.rgb/episode_{i:06}.mp4').exists()
    # Metadata also reads both chunks (not just the writer).
    summary=metadata(output,rows,['j']*29)
    assert summary['episodes']==2


def test_legacy_writers_never_return_unchecked_cache():
    from tools import study_joint_perturbations, study_elbow_correction, search_larger_augmentations, tune_larger_lifts
    for call in [lambda:study_joint_perturbations.prepare(0),lambda:study_elbow_correction.evaluate((0,.4,False)),
                 lambda:search_larger_augmentations.prepare(0,[0.,0.]),lambda:tune_larger_lifts.run((0,.001,200))]:
        with pytest.raises(RuntimeError,match='retired'):call()


def test_source_changed_before_cached_stage_return(monkeypatch,tmp_path):
    from g1_pipette_sim.augmentation import runner
    c={'folder':str(tmp_path),'run_id':'r'}
    atomic_json(tmp_path/'screen.json',{'result':{'pickup':True}})
    seal(tmp_path,'screen',runner.candidate_inputs(c),[tmp_path/'screen.json'])
    def changed(c):raise ValueError('Source changed')
    monkeypatch.setattr(runner,'load_episode',changed)
    with pytest.raises(ValueError,match='Source changed'):runner.stage(c)


def test_uncommitted_capture_recovers_only_with_matching_commit_hashes(monkeypatch,tmp_path):
    from g1_pipette_sim.augmentation import runner
    from tools import collect_pickup_augmentations as collector
    c={'folder':str(tmp_path),'run_id':'r'}
    np.savez(tmp_path/'trajectory.npz',timestamp=np.arange(3.))
    for name in ['verification.json','camera-poses.json','rgb.mp4','wrist_right.mp4','wrist_left.mp4']:
        (tmp_path/name).write_bytes(b'fixture')
    hashes={p.name:file_hash(p) for p in tmp_path.iterdir()}
    atomic_json(tmp_path/'accepted.json',{'candidate':c,'artifact_hashes':hashes,'validation':{'pickup':True,'frame_count':3}})
    monkeypatch.setattr(runner,'load_episode',lambda c:None)
    monkeypatch.setattr(collector,'audit_videos',lambda *a:None)
    monkeypatch.setattr(runner,'validate',lambda *a,**kw:pytest.fail('Must recover without another rollout'))
    assert runner.stage(c,True)['pickup']
    assert verify(tmp_path,'record',runner.candidate_inputs(c))


def test_export_rejects_stale_or_corrupt_publication(monkeypatch,tmp_path):
    from g1_pipette_sim.augmentation import export
    value={'entries':[],'duplicates':[]}
    monkeypatch.setattr(export,'dependencies',lambda:{})
    output=tmp_path/'published';output.mkdir();(output/'data').write_text('original')
    contract={'catalog':value,'dependencies':{},'v3':False}
    seal(output,'publication',contract,[output/'data'])
    assert export.publish(value,output)==0
    with pytest.raises(ValueError,match='Stale'):export.publish({**value,'revision':2},output)
    (output/'data').write_text('modified')
    with pytest.raises(ValueError,match='Corrupt'):export.publish(value,output)
