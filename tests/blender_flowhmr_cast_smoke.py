"""Exercise the actual Animate and Ground All Characters operator in Blender."""
import argparse
import json
from pathlib import Path
import sys
import time
import addon_utils
import bpy
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root/'blender_extension'))
addon_utils.enable('motius_flowhmr', default_set=True, persistent=False)
parser = argparse.ArgumentParser()
parser.add_argument('--motion', required=True)
parser.add_argument('--output', required=True)
a = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
props = bpy.context.scene.flowhmr
props.motion_file = str(Path(a.motion).resolve())
arms = [o for o in bpy.context.scene.objects if o.type == 'ARMATURE' and o.get('flowhmr_character')]
assert len(arms) == 3
started = time.perf_counter()
result = bpy.ops.flowhmr.animate_cast()
seconds = time.perf_counter()-started
assert result == {'FINISHED'}, props.message
assert props.state == 'completed'
reports = [json.loads(arm.animation_data.action['flowhmr_grounding_report']) for arm in arms]
assert all(report['frames'] == 750 for report in reports)
assert all(arm.animation_data.action.get('flowhmr_grounded') for arm in arms)
Path(a.output).write_text(json.dumps({'seconds':seconds, 'operator':'flowhmr.animate_cast',
    'frames':750, 'actors':[arm.name for arm in arms], 'reports':reports,
    'state':props.state}, indent=2))
print('BLENDER_CAST_OPERATOR_COMPLETE', seconds, flush=True)
