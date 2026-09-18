"""Compare strict rack-contact verdicts against immediate and original simulation sources."""
import json,csv,hashlib
from pathlib import Path
from collections import Counter,defaultdict
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/hil194/rack-contact-audit'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def report(p):
    d=read(p);r=d.get('report',d)
    pairs={tuple(sorted(p)) for p in r['contact_body_pairs']}
    hit=bool(r['contact_diagnostics'].get('pipette/rack',{}).get('samples',0))
    if hit!=(('pipette','rack') in pairs):raise ValueError(f'Inconsistent contact evidence: {p}')
    return hit or ('ejector','rack') in pairs

def category(source,child):
    return {(True,True):'contact_already_in_source',(False,True):'new_contact_after_augmentation',(True,False):'source_contact_absent_in_augmentation',(False,False):'both_contact_free'}[source,child]


def main():
    manifest=read(ROOT/'config/hil194-pickup-layouts.json');summaries={}
    for collection in ['original600','release544']:
        rows=[];sources={};roots={}
        for audited in csv.DictReader((OUT/collection/'episodes.csv').open()):
            folder=Path(audited['source_capture']);c=read(folder/'candidate.json');child_report=folder/'verification.json'
            if sha(child_report)!=audited['verification_sha256']:raise ValueError('Changed audited report')
            child=report(child_report);parent=c['parent_episode'];entry=manifest['episodes'][str(parent)];root_report=ROOT/entry['report'];root_cfg=ROOT/entry['config']
            source_folder=Path(c['source_capture']) if collection=='release544' else None
            source_report=source_folder/'verification.json' if source_folder else root_report
            # Original seeds pin the fitted parent configuration by hash.
            seed=read(source_folder/'candidate.json') if source_folder else c
            if seed['parent_episode']!=parent or seed['seed_config_sha256']!=entry['sha256'] or sha(root_cfg)!=entry['sha256']:raise ValueError('Changed or mismatched root seed')
            source=report(source_report);baseline=report(root_report)
            roots[parent]={'parent_episode':parent,'rack_contact':baseline,'report':str(root_report),'sha256':sha(root_report)}
            sources[str(source_report)]={'rack_contact':source,'sha256':sha(source_report)}
            rows.append({'dataset_episode':audited['dataset_episode'],'candidate_number':audited['candidate_number'],'capture_id':audited['capture_id'],'parent_episode':parent,'method':c['kind'],'classification':category(source,child),'root_classification':category(baseline,child),'source_rack_contact':source,'augmented_rack_contact':child,'original_baseline_rack_contact':baseline,'source_report':str(source_report),'augmented_report':str(child_report),'source_report_sha256':sha(source_report),'augmented_report_sha256':sha(child_report)})
        counts=Counter(r['classification'] for r in rows);methods=defaultdict(Counter)
        for r in rows:methods[r['method']][r['classification']]+=1
        failures=sum(r['augmented_rack_contact'] for r in rows)
        summary={'episodes':len(rows),'rack_contact_failures':failures,'transitions':dict(counts),'failure_share_already_in_source_percent':100*counts['contact_already_in_source']/failures,'failure_share_new_contact_percent':100*counts['new_contact_after_augmentation']/failures,'by_method':dict(methods),'unique_immediate_sources':len(sources),'original_baselines':list(roots.values()),'source_report_hashes':sources,'new_contact_episodes':[r for r in rows if r['classification']=='new_contact_after_augmentation'],'interpretation':'Contact already in source means the binary rack-contact failure predates this augmentation. It does not establish identical contact timing, geometry or severity. Source trajectories are simulated replays, not measured real-world contact labels.'}
        dest=OUT/collection
        (dest/'lineage-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        with (dest/'lineage-episodes.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        summaries[collection]=summary
        print(collection,json.dumps({'transitions':dict(counts),'by_method':dict(methods)}))
    (OUT/'lineage-summary.json').write_text(json.dumps(summaries,indent=2)+'\n')


if __name__=='__main__':main()
