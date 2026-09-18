# Third-party assets

`assets/robot/g1_inspire.xml` and `assets/robot/meshes/` were imported from the
G1/Inspire model in VLAPolicyBridge:
https://github.com/birbirll/VLAPolicyBridge

Original source path and per-mesh SHA-256 fingerprints are recorded in
`assets/robot/provenance.json`. These are third-party robot/hand assets; this
repository does not grant additional rights to them. No upstream license file
was present in the local VLAPolicyBridge checkout used for this import.

The DFQ hand geometry is a proxy for the recorded RH56DFTP hardware.

MuJoCo and other runtime libraries are installed separately; their respective
licenses apply. Real demonstration data is downloaded separately from its dataset
repository, not redistributed in this source repository.
