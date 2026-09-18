"""Copy the existing local robot and resolve its mesh paths; no downloads."""
from pathlib import Path
import argparse, shutil, xml.etree.ElementTree as ET, hashlib, json

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',type=Path,default=Path.home()/'Projects/VLAPolicyBridge/sim/decoupled_wbc/control/robot_model/model_data/g1/inspire/g1_29dof_rev_1_0_with_inspire_hand.xml')
args=parser.parse_args()
source=args.source.expanduser().resolve()
r=ET.parse(source).getroot()
destination=ROOT/'assets/robot';(destination/'meshes').mkdir(parents=True,exist_ok=True)
records=[]
for mesh in r.findall('./asset/mesh'):
    name=mesh.get('file')
    if not name: continue
    candidates=[source.parent/'meshes'/name,source.parent/r.find('compiler').get('meshdir','')/name,source.parent.parent/'meshes'/name]
    original=next((p for p in candidates if p.is_file()),None)
    if original is None: raise FileNotFoundError(f'Cannot resolve {name} from {source}')
    shutil.copy2(original,destination/'meshes'/Path(name).name)
    mesh.set('file',Path(name).name)
    records.append({'file':Path(name).name,'sha256':hashlib.sha256(original.read_bytes()).hexdigest()})
r.find('compiler').set('meshdir','meshes')
ET.indent(r)
ET.ElementTree(r).write(destination/'g1_inspire.xml',encoding='unicode')
(destination/'provenance.json').write_text(json.dumps({'source':str(source),'model':r.get('model'),'warning':'Local DFQ model is a proxy for the dataset RH56DFTP. Verify geometry, mounting and register mapping before transfer. Upstream asset terms remain applicable.','meshes':records},indent=2)+'\n')
print(f'Prepared {len(records)} meshes and {destination / "g1_inspire.xml"}')
