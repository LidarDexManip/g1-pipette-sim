"""Audit and summarize the complete 600-demonstration collection after export."""
import json,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.report_pickup_augmentations import report
from g1_pipette_sim.pickup_layouts import ROOT,sha256
from tools.fit_pickup_layouts import write_json

root=ROOT/'outputs/hil194/augmentation600'
while True:
 p=root/'export-progress.json'
 if p.exists():
  status=json.loads(p.read_text())
  if status.get('complete'):break
 time.sleep(30)
from tools.audit_augmented_dataset import audit
dataset_audit=audit(root)
summary=report(root)
if summary['accepted']!=status['target'] or summary['unique_geometry_action_hashes']!=status['target']:raise RuntimeError('Final accepted count mismatch')
text=(root/'report/REPORT.md').read_text()
for name in ['lerobot-v21','lerobot-v3']:
 for artifact in ['demonstrations.csv','coverage.png']:
  import shutil
  shutil.copy2(root/'report'/artifact,root/name/artifact)
 (root/name/'README.md').write_text(text+'\nSee meta/training-contract.json and meta/augmentation.jsonl for action semantics and parent grouping.\n')
write_json(root/'completion.json',{'complete':True,'accepted':summary['accepted'],'target':status['target'],'methods':summary['by_method'],'parents':summary['by_parent'],'distinct_wells':summary['distinct_wells'],'dataset_formats':['v2.1','v3.0'],'native_loader_checked':True,'all_episodes_audited':dataset_audit['episodes_checked'],'report_sha256':sha256(root/'report/summary.json'),'finished_at':time.time(),'scope':'Independent simulator validation; no real-world success claim.'})
print('COLLECTION AND AUDIT COMPLETE',summary['accepted'],flush=True)
