"""Summarize saved validation runs; no simulator or dataset mutation."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('outputs/calibration')
baseline=json.loads((root/'heldout-baseline/summary.json').read_text());final=json.loads((root/'final-calibrated/summary.json').read_text());base={r['episode']:r for r in baseline};base[53]=json.loads((root/'baseline/summary.json').read_text())[0]
held=[r for r in final if r['episode']!=53]
mean0=np.mean([base[r['episode']]['right_arm_rmse_rad'] for r in held]);mean1=np.mean([r['right_arm_rmse_rad'] for r in held]);print('Held-out mean',mean0,mean1,'reduction',1-mean1/mean0)
fig,ax=plt.subplots(figsize=(10,4));ids=[r['episode'] for r in final];x=np.arange(len(ids));ax.bar(x-.18,[base[i]['right_arm_rmse_rad'] for i in ids],.36,label='Original physics');ax.bar(x+.18,[r['right_arm_rmse_rad'] for r in final],.36,label='Identified gains + effective payload');ax.set_xticks(x,[f'{i}'+('\ntraining' if i==53 else '\nheld out') for i in ids]);ax.set_ylabel('Right-arm joint RMS error (radians)');ax.set_xlabel('Dataset episode');ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(root/'joint-error-comparison.png',dpi=170);plt.close(fig)
a=np.load(root/'final-calibrated/episode-053/trace.npz');b=np.load(root/'baseline/episode-053/trace.npz');fig,axes=plt.subplots(4,2,figsize=(12,10),sharex=True);names=['Shoulder pitch','Shoulder roll','Shoulder yaw','Elbow','Wrist roll','Wrist pitch','Wrist yaw']
for j,ax in enumerate(axes.flat[:7]):
 ax.plot(a['time'],a['observed'][:,22+j],color='black',label='Recorded state',lw=1.5);ax.plot(b['time'],b['q'][:,22+j],label='Original physics',alpha=.7);ax.plot(a['time'],a['q'][:,22+j],label='Calibrated physics',alpha=.8);ax.set_title(names[j]);ax.set_ylabel('radians');ax.grid(alpha=.2)
axes.flat[0].legend(fontsize=8);axes.flat[-1].axis('off');fig.supxlabel('Recorded time (seconds)');fig.tight_layout();fig.savefig(root/'episode53-joints.png',dpi=150);plt.close(fig)
table='| Episode | Original arm RMS (rad) | Calibrated arm RMS (rad) | Final simulated tip state |\n|---|---:|---:|---|\n'
for r in final:table+=f"| {r['episode']} | {base[r['episode']]['right_arm_rmse_rad']:.4f} | {r['right_arm_rmse_rad']:.4f} | {r['state']} |\n"
(root/'results-table.md').write_text(table)
(root/'aggregate.json').write_text(json.dumps({'heldout_episodes':[r['episode'] for r in held],'mean_baseline_right_arm_rmse_rad':float(mean0),'mean_calibrated_right_arm_rmse_rad':float(mean1),'relative_error_reduction':float(1-mean1/mean0),'all_numerically_clean':all(not r['warnings'] for r in final),'calibrated_disposal_count':sum(r['state']=='disposed' for r in final)},indent=2)+'\n')
print(table)
