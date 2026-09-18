"""Report only independently validated, recorded augmentation captures."""
import argparse,json
from pathlib import Path
from collections import Counter

def status(path):
 p=Path(path);s=json.loads((p/'progress.json').read_text());rows=[]
 for f in (p/'candidates').glob('*/accepted.json'):
  r=json.loads(f.read_text());c=r['candidate'];v=r['validation']
  rows.append({'candidate':c['number'],'parent':c['parent_episode'],'kind':c['kind'],'cell':c['cell'],'lift_mm':v['lift_m']*1000,'reference_mm':v['lift_target_m']*1000})
 result={'accepted':len(rows),'target':s['target'],'status':s['status'],'screened':s['screened'],'rejected':s['rejected'],'errors':s['errors'],'by_parent':dict(Counter(r['parent'] for r in rows)),'by_method':dict(Counter(r['kind'] for r in rows)),'distinct_wells':len({tuple(r['cell']) for r in rows}),'recent':sorted(rows,key=lambda r:r['candidate'])[-10:]}
 print(json.dumps(result,indent=2));return result

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('path',nargs='?',default='outputs/hil194/augmentation600');a=p.parse_args();status(a.path)
