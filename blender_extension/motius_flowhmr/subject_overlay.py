# SPDX-License-Identifier: GPL-3.0-or-later
"""Persist actual detector boxes as native Movie Clip tracking patterns."""
import bpy


def install_markers(context, data, selected):
    video=data.get('preview_sequence') or bpy.path.abspath(context.scene.flowhmr.video)
    clip=next((c for c in bpy.data.movieclips if bpy.path.abspath(c.filepath)==video),None)
    clip=clip or bpy.data.movieclips.load(video)
    clip.frame_start=1
    for track in clip.tracking.tracks:
        if track.name.startswith('FlowHMR person '):
            track.select = False
            for marker in track.markers:
                marker.mute = True
    # Detection is on the normalized 30fps sequence. A different source rate
    # needs that normalized video for frame-accurate overlays and playback.
    if not data.get('preview_sequence') and abs(clip.fps-30)>0.05:
        clip=bpy.data.movieclips.load(data['video'],check_existing=True)
    for item in data['tracks']:
        name = f'FlowHMR person {item["id"]}'
        track=clip.tracking.tracks.get(name) or clip.tracking.tracks.new(name=name,frame=1)
        track.use_custom_color=True;track.color=(.22,.83,.69) if item['id']==selected else (.8,.83,.86)
        observed=dict(zip(item['frames'],item['boxes']))
        for frame in range(data['frames']):
            box=observed.get(frame)
            marker=track.markers.find_frame(frame+1,exact=True) or track.markers.insert_frame(frame+1)
            marker.mute=box is None
            if box is not None:
                x1,y1,x2,y2=box;w,h=data['width'],data['height']
                marker.co=((x1+x2)/2/w,1-(y1+y2)/2/h)
                dx,dy=(x2-x1)/2/w,(y2-y1)/2/h
                marker.pattern_corners=((-dx,-dy),(dx,-dy),(dx,dy),(-dx,dy))
        if item['id']==selected:clip.tracking.tracks.active=track
        track.select=item['id']==selected
    for area in context.screen.areas:
        if area.type=='CLIP_EDITOR':
            space=area.spaces.active;space.clip=clip;space.show_marker_pattern=True;space.show_marker_search=False;space.show_track_path=False;space.show_names=True
