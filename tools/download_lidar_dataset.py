"""Download the pinned two-view G1 demonstrations used for LIDAR replay diagnostics."""
from pathlib import Path
from huggingface_hub import snapshot_download

REPO='jren313/g1-pipette-2view-teleop0925-5task-eerel'
REVISION='c7adfc179e5e15ade41b06803542001a24959dda'
ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    print(snapshot_download(REPO,repo_type='dataset',revision=REVISION,
                            cache_dir=str(ROOT/'.cache/huggingface/hub'),max_workers=4))
