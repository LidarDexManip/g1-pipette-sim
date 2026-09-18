"""Apply a strict no-pipette/rack-contact verdict to saved augmented rollouts.

Read-only with respect to captures and exports. Historical acceptance receipts
remain intact; this writes a separate strict-rule audit and exclusion lists.
"""
import csv,json,hashlib
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/hil194/rack-contact-audit'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def jsonl(path):return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit(name,entries):
    rows=[]
    for e in entries:
        folder=Path(e['source_capture']);path=folder/'verification.json';saved=json.loads(path.read_text());r=saved['report'];result=saved['result'];sha=digest(path)
        if e.get('validation_sha256') and sha!=e['validation_sha256']:raise ValueError(f'Changed verification: {path}')
        if not result['complete'] or not result['pickup'] or not r['finite']:raise ValueError(f'Not a complete accepted rollout: {path}')
        stats=r['contact_diagnostics'].get('pipette/rack',{});samples=stats.get('samples',0)
        pairs={tuple(sorted(pair)) for pair in r['contact_body_pairs']}
        if bool(samples)!=(('pipette','rack') in pairs):raise ValueError(f'Inconsistent contact records: {path}')
        # Check all available receipts, without interpreting timestep sample counts as contact duration.
        accepted=json.loads((folder/'accepted.json').read_text())
        if accepted['validation']['contacts'].get('pipette/rack',{})!=stats:raise ValueError(f'Accepted/report diagnostic disagreement: {path}')
        extra=any('rack' in pair and 'ejector' in pair for pair in pairs)
        failed=bool(samples or extra)
        rows.append({'dataset_episode':e.get('episode_index',''),'candidate_number':e['candidate_number'],'capture_id':e.get('capture_id',''),'parent_episode':e['parent_episode'],'split':e.get('split','unsplit'),'augmentation_method':accepted['candidate']['kind'],'strict_verdict':'failure_rack_contact' if failed else 'passes_rack_contact_check','pipette_rack_samples':samples,'maximum_normal_force_N':stats.get('maximum_normal_force_N',0),'maximum_penetration_mm':stats.get('maximum_penetration_mm',0),'ejector_rack_contact':extra,'source_capture':str(folder),'verification_sha256':sha})
    counts=Counter(r['strict_verdict'] for r in rows);failure=counts['failure_rack_contact'];by_split={}
    for split in sorted({r['split'] for r in rows}):
        subset=[r for r in rows if r['split']==split];n=sum(r['strict_verdict']=='failure_rack_contact' for r in subset)
        by_split[split]={'episodes':len(subset),'rack_contact_failures':n,'no_rack_contact':len(subset)-n}
    summary={'collection':name,'episodes':len(rows),'rack_contact_failures':failure,'no_rack_contact':len(rows)-failure,'failure_percent':100*failure/len(rows),'by_split':by_split,'rule':'Any recorded direct pipette/rack contact (or ejector/rack contact) fails; no positive force/penetration threshold. Tip/rack contact is allowed.','evidence':'Independent complete rollout verification reports; cumulative contacts sampled on every simulation step. Contacts also cross-checked against body-pair sets and acceptance receipts.','limitations':['Contact counts are contact-point samples, not unique impacts or seconds.','Reports do not save contact onset times or whether contact occurs before/after attachment.','No-rack-contact verdict is not proof of real-world success or absence of other failure modes.'],'source_reports_modified':False,'training_exports_modified':False}
    by_method=[]
    for method in sorted({r['augmentation_method'] for r in rows}):
        subset=[r for r in rows if r['augmentation_method']==method]
        n=sum(r['strict_verdict']=='failure_rack_contact' for r in subset)
        by_method.append({'method':method,'episodes':len(subset),'rack_contact_failures':n,
            'share_of_all_failures_percent':100*n/failure if failure else 0.,
            'within_method_failure_rate_percent':100*n/len(subset)})
    summary['by_method']=by_method
    dest=OUT/name;dest.mkdir(parents=True,exist_ok=True)
    (dest/'method-breakdown.json').write_text(json.dumps(by_method,indent=2)+'\n')
    for filename,subset in [('episodes.csv',rows),('failures.csv',[r for r in rows if r['strict_verdict']=='failure_rack_contact']),('no-rack-contact.csv',[r for r in rows if r['strict_verdict']!='failure_rack_contact'])]:
        with (dest/filename).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(subset)
    (dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main():
    collection=ROOT/'outputs/hil194/augmentation600';progress=json.loads((collection/'progress.json').read_text());entries=[]
    for n in progress['accepted']:
        f=collection/f'candidates/{n:06d}';c=json.loads((f/'candidate.json').read_text());entries.append({'candidate_number':n,'parent_episode':c['parent_episode'],'source_capture':str(f)})
    results=[audit('original600',entries)]
    release=ROOT/'outputs/hil194/augmentation544-ego-repair-v2b/release';entries=jsonl(release/'hub-upload/meta/augmentation.jsonl');split={}
    for label in ['sim_train','sim_val']:
        for r in jsonl(release/f'v21/{label}/meta/augmentation.jsonl'):split[r['capture_id']]=label
    for e in entries:e['split']=split[e['capture_id']]
    results.append(audit('release544',entries))
    (OUT/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    for r in results:print(json.dumps({k:r[k] for k in ['collection','episodes','rack_contact_failures','no_rack_contact','failure_percent','by_split']}))


if __name__=='__main__':main()
