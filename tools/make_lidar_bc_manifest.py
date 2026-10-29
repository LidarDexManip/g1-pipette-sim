"""Select ALL recorded pickup successes, splitting by recording session."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",required=True)
    p.add_argument("--validation-session",default="20260924_2016")
    p.add_argument("--output",required=True)
    args=p.parse_args()
    root=Path(args.dataset).expanduser().resolve()
    episodes={r["episode_index"]:r for r in map(json.loads,(root/"meta/episodes.jsonl").read_text().splitlines())}
    provenance=list(map(json.loads,(root/"meta/source_provenance.jsonl").read_text().splitlines()))
    goal="Pick up the pipette from the holder on the right with the right hand."
    source=dict(root=str(root),action_contract="g1_joint29_root6_hand_registers",command_alignment="paired_image_rows",
                register_max=1000,video_key="observation.images.rgb",idle_weight=.2,train=[],validation=[])
    for row in provenance:
        if goal not in episodes[row["episode_index"]]["tasks"] or row["source_label"]!="successful":continue
        split="validation" if row["source_session"]==args.validation_session else "train"
        source[split].append(dict(episode=row["episode_index"],session=row["source_session"],task="pipette_pickup"))
    if not source["train"] or not source["validation"]:raise ValueError("Empty session split")
    output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(datasets=[source]),indent=2)+"\n")
    print(json.dumps(dict(training_episodes=len(source["train"]),validation_episodes=len(source["validation"]))))


if __name__=="__main__":main()
