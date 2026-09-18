"""Summarize kinematic diagnostic videos and compare insertion-phase side views."""
import csv,json,hashlib
from pathlib import Path
import cv2
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/hil194/original-baseline-comparisons-kinematic-state'

def read(p):return json.loads(p.read_text())
def frame(path,index):
    c=cv2.VideoCapture(str(path));c.set(cv2.CAP_PROP_POS_FRAMES,index);ok,bgr=c.read();c.release()
    if not ok:raise ValueError(f'Cannot read {path}:{index}')
    return cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)

def main():
    progress=read(OUT/'progress.json');assert progress['complete'] and progress['completed']==12 and not progress['errors']
    rows=[];sheets=[];font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    titles=['Real side camera','Dynamic motor targets','Dynamic measured targets','Kinematic measured poses']
    for offset,episode in enumerate(progress['requested']):
        folder=OUT/f'episode-{episode:03d}';audit=read(folder/'highres-render.json');diag=read(folder/'kinematic-diagnostics.json');ref=read(folder/'report.json')
        assert audit['source']=='state' and audit['physics']=='kinematic' and audit['replay_report']['finite']
        assert hashlib.sha256((folder/'simulation.mp4').read_bytes()).hexdigest()==audit['video_sha256']
        parents=[ROOT/'outputs/hil194/original-baseline-comparisons'/folder.name,ROOT/'outputs/hil194/original-baseline-comparisons-state'/folder.name]
        target=read(parents[0]/'highres-render.json')['replay_report'];state=read(parents[1]/'highres-render.json')['replay_report']
        t=next(e['time'] for e in ref['events'] if e['state']=='attached');n=min(round(t*30),audit['frames_verified']-1)
        rows.append({'episode':episode,'dynamic_target_rack_contact':bool(target['contact_diagnostics'].get('pipette/rack',{}).get('samples',0)),'dynamic_state_rack_contact':bool(state['contact_diagnostics'].get('pipette/rack',{}).get('samples',0)),'kinematic_sampled_overlap':diag['overlap_frames']>0,'kinematic_overlap_frames':diag['overlap_frames'],'kinematic_max_penetration_mm':diag['maximum_sampled_penetration_mm'],'kinematic_max_joint_error_rad':diag['max_body_joint_error_rad'],'frames':audit['frames_verified'],'review_time_s':n/30})
        if offset%4==0:
            sheet=Image.new('RGB',(1280,1440),'#14202c');draw=ImageDraw.Draw(sheet)
            for col,title in enumerate(titles):draw.text((col*320+5,5),title,font=font,fill='white')
            sheets.append(sheet)
        latest=frame(folder/'simulation.mp4',n);low=frame(parents[0]/'simulation.mp4',n);dyn=frame(parents[1]/'simulation.mp4',n)
        crops=[latest[:1080,2160:3240],low[1080:2160,2160:3240],dyn[1080:2160,2160:3240],latest[1080:2160,2160:3240]]
        y=40+(offset%4)*350
        draw=ImageDraw.Draw(sheets[-1]);draw.text((5,y),f'Episode {episode} | t={n/30:.2f}s (target-driven attachment time)',font=font,fill='#83d5d9')
        for col,img in enumerate(crops):sheets[-1].paste(Image.fromarray(img).resize((320,320)),(col*320,y+25))
    for i,sheet in enumerate(sheets):sheet.save(OUT/f'mode-comparison-{i+1}.jpg',quality=90)
    with (OUT/'diagnostic-summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={'complete':True,'episodes':12,'frames_verified':sum(r['frames'] for r in rows),'resolution_wh':[3240,2160],'fps':30,'maximum_body_joint_error_rad':max(r['kinematic_max_joint_error_rad'] for r in rows),'dynamic_target_contact_episodes':[r['episode'] for r in rows if r['dynamic_target_rack_contact']],'dynamic_state_contact_episodes':[r['episode'] for r in rows if r['dynamic_state_rack_contact']],'kinematic_overlap_episodes':[r['episode'] for r in rows if r['kinematic_sampled_overlap']],'kinematic_no_sampled_overlap_episodes':[r['episode'] for r in rows if not r['kinematic_sampled_overlap']],'not_comparable_success_metrics':'Dynamic contact is checked at every physics step; kinematic overlap is a 30 Hz geometric distance diagnostic, not force-based validation.','real_alignment':'Visual comparison sheets only; no independent real-image landmark-error measurement.','rows':rows}
    (OUT/'diagnostic-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
