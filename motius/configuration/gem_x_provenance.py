"""Released GEM-X provenance; inference lives in the native Pipeline."""
from pathlib import Path
from motius.utils.hashing import sha256_file
SOURCE_REPOSITORY = "https://github.com/NVlabs/GEM-X"
SOURCE_REVISION = "32992550dba114c62243fb55e361311972dce8f9"
HF_REPOSITORY = "nvidia/GEM-X"
HF_REVISION = "5ccf5ca3746c3620aa4016114f069a5f6ae399cd"
CHECKPOINT_FILENAME = "gem_soma.ckpt"
CHECKPOINT_SHA256 = "4c1f85ca8c1e11e6588aead49fbc024bf660708def670043e0b537c101ee298e"
GEM_X_REPO_ID = "ZeyuLing/Motius-GEM-X"
GEM_X_ARTIFACT_FORMAT = "motius-gem-x-v2"
OFFICIAL_OUTPUT_FILENAME = "hpe_results.pt"

def verify_checkpoint(path):
    path=Path(path).expanduser().resolve()
    if sha256_file(path)!=CHECKPOINT_SHA256:raise ValueError("GEM-X released checkpoint SHA256 mismatch")
    return path

def expected_output_path(video,output_root):
    return Path(output_root).expanduser().resolve()/Path(video).stem/OFFICIAL_OUTPUT_FILENAME

SOMA_SOURCE_REVISION="e0f8ff0ecfa3edbbb6058b1e0f08822ee2f84ee5"
