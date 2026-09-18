"""Create a read-only illustration index of accepted captures without editing run progress."""
import argparse
import json
from pathlib import Path

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection', type=Path, required=True)
    a = p.parse_args()
    accepted = []
    for path in sorted((a.collection/'candidates').glob('*/accepted.json')):
        record = json.loads(path.read_text())
        if record.get('independently_replayed') and record['validation']['pickup']:
            accepted.append(int(path.parent.name))
    target = a.collection/'illustration-index.json'
    target.write_text(json.dumps({'accepted':accepted}, indent=2)+'\n')
    print(f'{len(accepted)} accepted captures indexed in {target}')
