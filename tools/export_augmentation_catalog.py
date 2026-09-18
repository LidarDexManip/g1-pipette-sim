"""Build an immutable deduplicated catalog and optionally publish a dataset."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from g1_pipette_sim.augmentation.export import catalog,publish,verify_catalog
from g1_pipette_sim.augmentation.contracts import atomic_json

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collections',type=Path,nargs='+')
    p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--v3',action='store_true')
    a=p.parse_args()
    if a.catalog.exists():
        value=json.loads(a.catalog.read_text());verify_catalog(value)
        if a.collections and catalog(a.collections)!=value:raise ValueError('Catalog changed; use a new manifest path')
    else:
        if not a.collections:p.error('--collections required to create catalog')
        value=catalog(a.collections);atomic_json(a.catalog,value)
    if a.output:publish(value,a.output,a.v3)
    print(json.dumps({'unique':len(value['entries']),'duplicates':len(value['duplicates'])}))
