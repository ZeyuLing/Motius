# ProtoMotions Attribution

Native neural modules, G1 environment components, reference-motion data, and PPO/AMP objectives are derived from NVlabs/ProtoMotions commit `49fe5ad69de67ebbc07ea2b25d41b0f622c15c3c`, under Apache-2.0. The native Trainer runs through the common Motius AccelerateRunner. Original application and trainer packages are not included or imported.

The network directory contains neural modules and their Bundle. Observation encoding, action conversion, physics, reference data, rollout collection, and training objectives reside in their respective framework components. `environment_source_map.json` records the selected environment modules and native destinations. Full upstream license and third-party notices are preserved here.

`g1-assets.json` records the exact sizes and hashes of the 75 selected G1 mesh, MJCF and USD files. Run `python tools/download_protomotions_assets.py` to install verified assets under `checkpoints/robots/protomotions`; binary robot assets are kept outside the Python model package. Isaac Sim and Isaac Lab are separately installed runtime dependencies.
