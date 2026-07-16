# Mixamo Character Setup

Adobe-provided Mixamo characters are not redistributed or replaced by
look-alike built-ins. To use a specific character:

1. Sign in at [Mixamo](https://www.mixamo.com/), select a character, and
   download a rigged FBX with skin.
2. Store the file at the stable path below.
3. Pass the path to `export_motion_to_fbx`. Standard `mixamorig:*` bone names
   are detected automatically; add `bone_map.json` only for a custom rig.

```text
checkpoints/characters/mixamo/
└── x_bot/
    ├── character.fbx
    └── bone_map.json       # optional
```

```python
from motius.motion import export_motion_to_fbx

export_motion_to_fbx(
    motion,
    source_representation="hml263",
    character_fbx="checkpoints/characters/mixamo/x_bot/character.fbx",
    output_path="outputs/fbx/x_bot.fbx",
    model_path="checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl",
    backend="fbxsdk",
)
```

Files downloaded from Mixamo remain local and are ignored by Git. Review the
official [Mixamo FAQ](https://helpx.adobe.com/creative-cloud/faq/mixamo-faq.html)
and [additional terms](https://wwwimages2.adobe.com/content/dam/cc/en/legal/servicetou/Mixamo-Addl-Terms-en_US-20210623.pdf)
for the assets obtained from that service.
