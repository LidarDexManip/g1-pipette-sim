# Red tape on the pipette

The three side-by-side photos `20260910_205902275_iOS.heic`, `20260910_205920189_iOS.heic` and `20260910_210014289_iOS.heic` show the same pipette shaft, wrapped in red tape immediately above the black nozzle. It is not an extended nozzle or a new tip adapter.

The mesh uses a tapered red skin following the existing blue shaft profile. Axial coverage is approximately 20 mm, estimated against the exposed 12.7 mm nozzle. Its lower edge begins at the blue/black junction. A 20-micron visual offset prevents z-fighting; this is not a measured tape thickness. The bare nozzle, insertion depth, collision shapes and 80 g assembly mass remain unchanged.

The HIL194 September 9 session (episodes 46–193) enables this skin. The older session remains untaped to match its images. The former red nozzle-reference marker stays invisible.

```bash
./run.sh --episode 46
~/miniconda3/envs/lerobot-act/bin/python tools/export_pipette_mesh.py \
  --episode 46 --output outputs/hil194/red-tape/mesh
```

The export contains `pipette.obj` and its required `pipette.mtl`. The inspection render is `outputs/hil194/red-tape/preview.png`; source-photo comparison is `outputs/hil194/red-tape/references.jpg`.
