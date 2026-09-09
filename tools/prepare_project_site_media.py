#!/usr/bin/env python3
"""Derive posters and a browser MP4 from existing, published demo frames.

No pose, camera, color or timing edits; posters are ordinary video thumbnails.
Run once when source demos change; the Pages build only copies these outputs.
"""
from pathlib import Path
import subprocess
import json
import hashlib
import imageio_ffmpeg
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

def main():
    sources = {
        'generation': 'assets/model_zoo/hymotion_t2m/hymotion_t2m_full_humanml3d_004545_smpl_mesh_512_30fps.gif',
        'dance': 'assets/model_zoo/bailando/bailando_aistpp_waacking_gWA_mWA0_smpl_mesh_512_30fps.gif',
        'autorig': 'assets/motion/auto_rigging_demo/motius_multi_character_autorig_004822_800x450_20fps.gif',
        'characters': 'assets/motion/fbx_character_demo/004822_skeleton_smpl_mixamo_1440_30fps.gif',
        'representations': 'assets/motion/representation_demo/004822_hml_smpl_soma_core_g1_1920_30fps.gif',
    }
    output = ROOT / 'site/media'
    output.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for name, source in sources.items():
        path = ROOT / source
        with Image.open(path) as im:
            poster_frame = min(30, im.n_frames-1)
            im.seek(poster_frame)
            poster = im.convert('RGB')
            poster.thumbnail((1440, 900))
            poster.save(output / f'{name}.jpg', quality=88)
        provenance[name] = {'source': source, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'poster_frame': poster_frame}
    conversions = {
        'representations': (ROOT / sources['representations']).with_suffix('.mp4'),
        'generation': output / 'generation.mp4',
    }
    for name, target in conversions.items():
        source = ROOT / sources[name]
        if not target.exists():
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error',
                            '-n', '-i', str(source), '-fps_mode', 'vfr', '-c:v', 'libx264',
                            '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(target)], check=True)
        provenance[name]['derived_video'] = target.relative_to(ROOT).as_posix()
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n', encoding='utf-8')

if __name__ == '__main__':
    main()
