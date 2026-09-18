"""Read-only storage accounting, including hard-link-aware reclaim estimates."""
import os
from pathlib import Path
import shutil
from .contracts import atomic_json


def category(path):
    name=path.name
    if path.suffix in ['.mp4','.avi','.mov']:return 'videos'
    if name=='commands.npz' or name=='rejected-commands.npz':return 'replay_commands'
    if name=='trajectory.npz':return 'accepted_states'
    if path.suffix=='.npz':return 'diagnostic_arrays'
    if path.suffix=='.parquet':return 'training_tables'
    if path.suffix in ['.png','.jpg','.jpeg']:return 'images'
    if path.suffix in ['.json','.jsonl','.csv','.sqlite3','.md']:return 'manifests_reports_and_ledger'
    return 'other'


def inventory(root, group_depth=1):
    root=Path(root).resolve(); groups={};inodes={};stack=[root]
    while stack:
        directory=stack.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):stack.append(Path(entry.path));continue
                stat=entry.stat(follow_symlinks=False);path=Path(entry.path)
                relative=path.relative_to(root);group='/'.join(relative.parts[:group_depth]) if len(relative.parts)>group_depth else '(root files)'
                row=groups.setdefault(group,{'logical_bytes':0,'allocated_bytes':0,'file_count':0,'estimated_reclaimable_bytes':0,'categories':{}})
                cat=row['categories'].setdefault(category(path),{'logical_bytes':0,'file_count':0})
                row['logical_bytes']+=stat.st_size;row['file_count']+=1;cat['logical_bytes']+=stat.st_size;cat['file_count']+=1
                key=(stat.st_dev,stat.st_ino)
                inode=inodes.setdefault(key,{'blocks':stat.st_blocks*512,'logical':stat.st_size,'links':stat.st_nlink,'groups':{},'seen':0})
                inode['groups'][group]=inode['groups'].get(group,0)+1;inode['seen']+=1
    for inode in inodes.values():
        for group,count in inode['groups'].items():
            groups[group]['allocated_bytes']+=inode['blocks']
            if count==inode['links']:groups[group]['estimated_reclaimable_bytes']+=inode['blocks']
    disk=shutil.disk_usage(root)
    return {'root':str(root),'groups':groups,'logical_bytes':sum(r['logical_bytes'] for r in groups.values()),
            'unique_inode_allocated_bytes':sum(r['blocks'] for r in inodes.values()),
            'estimated_reclaimable_bytes_if_entire_root_removed':sum(r['blocks'] for r in inodes.values() if r['seen']==r['links']),
            'filesystem_free_bytes':disk.free,'filesystem_total_bytes':disk.total,
            'notes':['Read-only snapshot; active files may grow during traversal.',
                     'Logical bytes count each pathname; unique allocated bytes count each inode once.',
                     'Per-group reclaim estimates exclude inodes with links elsewhere; filesystem snapshots/reflinks can reduce actual freed bytes.',
                     'Do not delete manifests, source captures, command archives, or diagnostic states if replay/clustering must remain possible.']}


def write_storage(root, output, group_depth=1):
    value=inventory(root,group_depth);atomic_json(output,value)
    lines=['# Storage inventory','',f"Root: `{value['root']}`",'',
           '| Group | Logical GiB | Allocated GiB | Estimated reclaimable GiB | Files |',
           '| --- | ---: | ---: | ---: | ---: |']
    for name,row in sorted(value['groups'].items(),key=lambda item:-item[1]['allocated_bytes']):
        lines.append(f"| {name} | {row['logical_bytes']/2**30:.3f} | {row['allocated_bytes']/2**30:.3f} | {row['estimated_reclaimable_bytes']/2**30:.3f} | {row['file_count']} |")
    lines+=['',f"Unique allocated total: {value['unique_inode_allocated_bytes']/2**30:.3f} GiB. Filesystem free: {value['filesystem_free_bytes']/2**30:.1f} GiB.",'',*value['notes']]
    Path(output).with_suffix('.md').write_text('\n'.join(lines)+'\n')
    return value
