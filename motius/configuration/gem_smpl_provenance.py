"""Released GEM-SMPL provenance; inference lives in the native Pipeline."""
from pathlib import Path
from motius.utils.hashing import sha256_file
SOURCE_REPOSITORY = "https://github.com/NVlabs/GENMO"
SOURCE_REVISION = "16bebf402d8893184249ee206d957b8248cd8310"
HF_REPOSITORY = "nvidia/GEM-X"
HF_REVISION = "5ccf5ca3746c3620aa4016114f069a5f6ae399cd"
CHECKPOINT_FILENAME = "gem_smpl.ckpt"
CHECKPOINT_SHA256 = "1d15cbe2864d6de61a75e83fdbfe83bec3c7b183eee3d3dcdbd9107e4456454a"
GEM_SMPL_REPO_ID = "ZeyuLing/Motius-GEM-SMPL"
GEM_SMPL_ARTIFACT_FORMAT = "motius-gem-smpl-v2"
OFFICIAL_OUTPUT_FILENAME = "smpl_params.pt"

def verify_checkpoint(path):
    path=Path(path).expanduser().resolve()
    if sha256_file(path)!=CHECKPOINT_SHA256:raise ValueError("GEM-SMPL released checkpoint SHA256 mismatch")
    return path

def expected_output_path(video,output_root):
    return Path(output_root).expanduser().resolve()/Path(video).stem/OFFICIAL_OUTPUT_FILENAME
