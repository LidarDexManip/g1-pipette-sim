import json,os
from pathlib import Path
import numpy as np
from g1_pipette_sim.augmentation.campaign import schedule
from g1_pipette_sim.augmentation.contracts import atomic_json
from g1_pipette_sim.augmentation.store import Ledger
from g1_pipette_sim.augmentation.campaign_report import report,clustering_export
from g1_pipette_sim.augmentation.storage import inventory


def sources():
    return [{'parent_episode':i,'folder':f'/source/{i}','candidate':{},'partition':'train','source_video':f'/video/{i}',
             'config':{'rack':{'active_cell':[0,0],'rows':8,'cols':12}},
             'phase':{'eligible':i!=45,'approach_s':[1,2,4] if i!=45 else None,'reason':'short' if i==45 else None}}
            for i in [6,14,45]]


def test_balanced_schedule_and_paired_directions(tmp_path):
    specs=schedule(sources(),42,tmp_path,'id')
    assert len(specs)==72 and len({s['id'] for s in specs})==72
    for parent in [6,14,45]:
        for family in ['placement_only','joint_approach']:
            chosen=[s for s in specs if s['parent_episode']==parent and s['family']==family]
            assert len(chosen)==12
            if family=='joint_approach':
                for i in range(6):assert chosen[i]['recipe']['direction']==chosen[i+6]['recipe']['direction']
            else:
                assert len({tuple(s['recipe']['cell']) for s in chosen})==12
                assert all(s['recipe']['method']=='well_only' for s in chosen)
    assert sum(not s['eligible'] for s in specs)==12
    assert specs==schedule(sources(),42,tmp_path,'id')


def test_failures_keep_parent_attribution_without_candidate(tmp_path):
    specs=schedule(sources(),42,tmp_path,'id')[:6];ledger=Ledger(tmp_path)
    for i,spec in enumerate(specs):
        ledger.add(spec['id']);ledger.finish(spec['id'],'skipped' if not spec['eligible'] else 'rejected',{'generation_rejected':spec['eligible'],'reason':'bounds'})
    value,attempts=report(tmp_path,specs,ledger)
    assert value['status']=='complete';assert value['totals']['skipped']==1
    assert value['totals']['generation_rejected']==5;assert not attempts
    assert len(value['by_original_and_family'])==6
    assert 'bounds' in (tmp_path/'reports/proposals.csv').read_text()
    ledger.close()


def test_clustering_masks_truncated_failure_and_keeps_labels_separate(tmp_path):
    folder=tmp_path/'capture';folder.mkdir();(tmp_path/'reports').mkdir()
    atomic_json(folder/'candidate.json',{'seed_attachment_s':5.})
    np.savez(folder/'screen-state.npz',time_s=np.array([0.,1.,2.,3.]),nozzle_relative_tip_m=np.ones((4,3))*.002,
             angle_deg=np.ones(4),maximum_contact_force_N=np.arange(4),maximum_penetration_mm=np.zeros(4))
    clustering_export(tmp_path,[{'folder':str(folder),'pickup':False,'reason':'no_attachment','original_episode':14}])
    with np.load(tmp_path/'reports/clustering-features.npz') as d:
        assert d['features'].shape==(1,64,6)
        assert d['valid'][0,0].all() and not d['valid'][0,-1].any()
        assert np.isnan(d['features'][0,-1]).all()
        assert np.allclose(d['features'][0,0,:3],2.)
    schema=json.loads((tmp_path/'reports/clustering-schema.json').read_text())
    assert 'original_episode' not in schema['feature_channels']


def test_storage_counts_hard_links_without_overstating_reclaim(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';a.mkdir();b.mkdir()
    (a/'one.npz').write_bytes(b'x'*10000);os.link(a/'one.npz',b/'two.npz')
    info=inventory(tmp_path)
    assert info['logical_bytes']==20000
    assert info['groups']['a']['estimated_reclaimable_bytes']==0
    assert info['groups']['b']['estimated_reclaimable_bytes']==0
    assert info['estimated_reclaimable_bytes_if_entire_root_removed']==info['unique_inode_allocated_bytes']
    assert (a/'one.npz').exists() and (b/'two.npz').exists()


def test_extended_schedule_uses_new_directions_and_cells(tmp_path):
    specs=schedule(sources(),43,tmp_path,'extended',24)
    joints=[s['recipe'] for s in specs if s['parent_episode']==6 and s['family']=='joint_approach']
    assert len(joints)==24
    assert joints[0]['direction']!=joints[12]['direction']
    for start in [0,12]:
        for i in range(6):
            assert joints[start+i]['direction']==joints[start+i+6]['direction']
            assert joints[start+i]['amplitude_deg']==1.
            assert joints[start+i+6]['amplitude_deg']==3.
    placement=[s['recipe'] for s in specs if s['parent_episode']==6 and s['family']=='placement_only']
    assert len({tuple(s['cell']) for s in placement})==24
    assert specs==schedule(sources(),43,tmp_path,'extended',24)
