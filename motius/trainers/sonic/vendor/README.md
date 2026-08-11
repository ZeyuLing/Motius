# SONIC training source snapshot

This directory contains the Apache-2.0 SONIC training source from
[`NVlabs/GR00T-WholeBodyControl`](https://github.com/NVlabs/GR00T-WholeBodyControl)
at commit `4141c34280abb67c82e115342a8720f4a83d750d`.

Motius invokes this snapshot through `python -m motius.trainers.sonic.train`.
No external source checkout is imported at runtime. Model weights remain under
the NVIDIA Open Model License; see `UPSTREAM_LICENSE`.

The checked-in asset set contains the Unitree G1 description and meshes used
by the public `sonic_release` training configuration. Additional H2 and legacy
robot asset families from the upstream repository are outside this public
configuration and are not packaged.
