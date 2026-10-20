"""Resolve implicit launcher defaults without substituting explicit user inputs."""
import json
from pathlib import Path
from .dataset import DEFAULT_DATASET, resolve_dataset_root

ROOT=Path(__file__).resolve().parents[1]
HIL293_DATASET=Path.home()/'.cache/huggingface/hub/datasets--jren313--g1-pipette-3view-hil293'


def configure_default_replay(args, root=ROOT, preferred=DEFAULT_DATASET, fallback=HIL293_DATASET):
    """Keep installed HIL194 defaults; use a dataset-bound HIL293 recipe if absent.

    An existing but ambiguous/incomplete cache must still report its error. Never
    replace an explicitly requested dataset or config with another demonstration.
    """
    if args.config is not None or args.dataset is not None or Path(preferred).exists():return None
    try:dataset=resolve_dataset_root(fallback)
    except FileNotFoundError as error:
        raise FileNotFoundError(f'Default HIL194 dataset is missing and HIL293 is unavailable. Pass --dataset /path/to/dataset --config /path/to/matching/workspace.json. {error}') from error
    index=11 if args.episode is None else args.episode
    folder=Path(root)/f'outputs/hil293/fixed-height-sweep/episode-{index:03d}'
    config=folder/'validated-config.json'
    label='saved pickup placement (historical validation; not a new success guarantee)'
    if not config.is_file():
        config=Path(root)/f'config/replay/hil293-episode-{index:03d}.json'
    if not config.is_file():
        config=Path(root)/f'outputs/hil293/baselines/episode-{index:03d}/workspace.json'
        label='unvalidated initial placement'
    if not config.is_file():
        raise FileNotFoundError(f'HIL293 is available but no matching config exists for episode {index}. Pass --config with its workspace, or omit --episode to open episode 11.')
    cfg=json.loads(config.read_text());record=cfg.get('replay_audit_setup',{})
    recorded=Path(record.get('dataset',''))
    if record.get('episode')!=index or recorded.name!=dataset.name or recorded.parent.parent.name!=Path(fallback).name:
        raise ValueError(f'HIL293 config does not match episode {index} and snapshot {dataset.name}: {config}')
    args.dataset=dataset;args.config=config;args.episode=index
    if args.scene_layout=='auto':args.scene_layout='fixed'
    return f'HIL194 is not cached; using HIL293 episode {index}, {label}: {config}'
