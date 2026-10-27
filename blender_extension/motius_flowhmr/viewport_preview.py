# SPDX-License-Identifier: GPL-3.0-or-later
"""Viewport-only playblast cache. Motion data and editable rigs remain unchanged."""
import bpy
import blf
import gpu
from mathutils import Matrix
from pathlib import Path
from .subjects import load_subjects, candidates, digest
from bpy.app.handlers import persistent
from bpy.types import Operator
from gpu_extras.batch import batch_for_shader

_CACHE = None
_BUILD = None


def clear():
    global _CACHE
    if _CACHE is not None:
        for image in _CACHE.get('images',[]):
            bpy.data.images.remove(image)
    _CACHE = None


def copy_texture(source, width, height):
    """GPU-to-GPU copy avoids interpreting RGBA8 byte storage as float pixels."""
    texture=gpu.types.GPUTexture((width,height),format='RGBA8')
    framebuffer=gpu.types.GPUFrameBuffer(color_slots=texture)
    viewport=gpu.state.viewport_get()
    try:
        with framebuffer.bind(), gpu.matrix.push_pop(), gpu.matrix.push_pop_projection():
            gpu.state.viewport_set(0,0,width,height)
            gpu.state.depth_test_set('NONE');gpu.state.blend_set('NONE')
            gpu.matrix.load_matrix(Matrix.Identity(4));gpu.matrix.load_projection_matrix(Matrix.Identity(4))
            shader=gpu.shader.from_builtin('IMAGE')
            batch=batch_for_shader(shader,'TRI_FAN',{'pos':[(-1,-1),(1,-1),(1,1),(-1,1)],'texCoord':[(0,0),(1,0),(1,1),(0,1)]})
            shader.bind();shader.uniform_sampler('image',source);batch.draw(shader)
    finally:
        gpu.state.viewport_set(*viewport)
    return texture


def key(context):
    s = context.scene
    v = context.space_data.region_3d
    return (s.as_pointer(), s.frame_start, s.frame_end, s.flowhmr.video, s.flowhmr.subjects_file, s.flowhmr.subject_id, s.flowhmr.show_surface,
            tuple((o.name,o.hide_viewport,o.hide_get()) for o in s.objects),
            tuple(x for row in v.view_matrix for x in row),
            tuple(x for row in v.window_matrix for x in row))


def ready(context):
    if context.area.type != 'VIEW_3D': return False
    return _CACHE is not None and _CACHE['key'] == key(context)


def draw(frame):
    if _CACHE is None:
        return
    texture = _CACHE['frames'][frame-_CACHE['start']]
    region = bpy.context.region
    shader = gpu.shader.from_builtin('IMAGE')
    batch = batch_for_shader(shader, 'TRI_FAN', {
        'pos': [(0,0),(region.width,0),(region.width,region.height),(0,region.height)],
        'texCoord': [(0,0),(1,0),(1,1),(0,1)]})
    shader.bind();shader.uniform_sampler('image',texture);batch.draw(shader)
    blf.position(0,20,region.height-65,0);blf.size(0,16);blf.color(0,.9,.95,.95,1)
    scene=bpy.context.scene
    blf.draw(0,f'Cached viewport · {scene.render.fps/scene.render.fps_base:g} fps · {frame-_CACHE["start"]+1} / {len(_CACHE["frames"])}')


def draw_source(frame):
    if _CACHE is None or not _CACHE['source_frames']:
        return
    index=frame-_CACHE['start']
    texture=_CACHE['source_frames'][index]
    region=bpy.context.region
    data=_CACHE['subjects'];w,h=data['width'],data['height']
    scale=min(region.width/w,(region.height-50)/h)
    x=(region.width-w*scale)/2;y=(region.height-h*scale)/2
    shader=gpu.shader.from_builtin('IMAGE')
    batch=batch_for_shader(shader,'TRI_FAN',{'pos':[(x,y),(x+w*scale,y),(x+w*scale,y+h*scale),(x,y+h*scale)],'texCoord':[(0,0),(1,0),(1,1),(0,1)]})
    shader.bind();shader.uniform_sampler('image',texture);batch.draw(shader)
    line=gpu.shader.from_builtin('UNIFORM_COLOR')
    for tid,box in candidates(data,frame-bpy.context.scene.flowhmr.start_frame):
        a,b,c,d=box
        points=[(x+a*scale,y+(h-d)*scale),(x+c*scale,y+(h-d)*scale),(x+c*scale,y+(h-b)*scale),(x+a*scale,y+(h-b)*scale)]
        batch=batch_for_shader(line,'LINE_LOOP',{'pos':points});line.bind();line.uniform_float('color',(.22,.83,.69,1) if tid==bpy.context.scene.flowhmr.subject_id else (.8,.8,.8,1));batch.draw(line)
    blf.position(0,x+10,y-25,0);blf.size(0,15);blf.color(0,.9,.95,.95,1);blf.draw(0,f'Source · frame {frame}')


class FLOWHMR_OT_prepare_playback(Operator):
    bl_idname = 'flowhmr.prepare_playback'
    bl_label = 'Prepare Original-Speed Playback'
    bl_description = 'Cache all viewport frames before playing at the original frame rate; Escape cancels'

    def execute(self, context):
        global _BUILD
        if _BUILD is not None:
            return {'CANCELLED'}
        clear()
        self.scene = context.scene
        self.area = context.area
        self.region = next(r for r in self.area.regions if r.type == 'WINDOW')
        self.space = self.area.spaces.active
        self.original_frame = self.scene.frame_current
        subjects=load_subjects(bpy.path.abspath(self.scene.flowhmr.subjects_file)) if self.scene.flowhmr.subjects_file else None
        has_source=any(a.type=='CLIP_EDITOR' and a.spaces.active.clip for a in context.screen.areas)
        if has_source and (not subjects or not subjects.get('preview_sequence')):
            self.report({'ERROR'},'Detect people first to prepare synchronized source playback')
            return {'CANCELLED'}
        if subjects and subjects['video_sha256'] != digest(bpy.path.abspath(self.scene.flowhmr.video)):
            self.report({'ERROR'},'Video changed; detect people again')
            return {'CANCELLED'}
        self.sidebar = self.space.show_region_ui
        self.space.show_region_ui = False
        self.next_frame = self.scene.frame_start
        self.frames = []
        self.source_frames=[]
        self.images=[]
        self.subjects=subjects
        self.offscreen = None
        self.timer = context.window_manager.event_timer_add(.001, window=context.window)
        context.window_manager.modal_handler_add(self)
        _BUILD = self
        self.scene.flowhmr.state = 'preparing playback'
        return {'RUNNING_MODAL'}

    def finish(self, context, success):
        global _BUILD, _CACHE
        context.window_manager.event_timer_remove(self.timer)
        if self.offscreen is not None:
            self.offscreen.free()
        self.scene.frame_set(self.scene.frame_start if success else self.original_frame)
        if success:
            _CACHE = dict(key=key(context), frames=self.frames, start=self.scene.frame_start,source_frames=self.source_frames,images=self.images,subjects=self.subjects,last_frame=self.scene.frame_current)
        else:
            self.frames.clear()
            for image in self.images:bpy.data.images.remove(image)
            self.space.show_region_ui = self.sidebar
        self.scene.flowhmr.state = 'completed' if success else 'cancelled'
        self.area.header_text_set(None)
        _BUILD = None
        if success:
            self.scene.flowhmr.message = 'Viewport cache ready · original-speed playback'
            bpy.ops.flowhmr.play_all()

    def modal(self, context, event):
        if _BUILD is not self:return {'CANCELLED'}
        if event.type == 'ESC' and event.value == 'PRESS':
            self.finish(context, False)
            return {'CANCELLED'}
        if event.type != 'TIMER':
            return {'RUNNING_MODAL'}
        try:
            if self.offscreen is None:
                self.width = min(960, self.region.width)
                self.height = round(self.region.height*self.width/self.region.width)
                source_width=min(640,self.subjects['width']) if self.subjects else 0
                source_height=round(source_width*self.subjects['height']/self.subjects['width']) if self.subjects else 0
                estimated=(self.scene.frame_end-self.scene.frame_start+1)*(self.width*self.height+source_width*source_height)*4
                if estimated>3*1024**3:
                    raise ValueError('Preview cache exceeds 3 GiB. Shorten the playback range or shrink the viewport.')
                self.offscreen = gpu.types.GPUOffScreen(self.width,self.height,format='RGBA8')
            if self.next_frame > self.scene.frame_end:
                self.finish(context,True)
                return {'FINISHED'}
            self.scene.frame_set(self.next_frame)
            if self.subjects and self.subjects.get('preview_sequence'):
                source=Path(self.subjects['preview_sequence']).parent/f'{self.next_frame-self.scene.flowhmr.start_frame:06d}.jpg'
                image=bpy.data.images.load(str(source),check_existing=False)
                try:
                    width=min(640,image.size[0]);height=round(image.size[1]*width/image.size[0])
                    self.source_frames.append(copy_texture(gpu.texture.from_image(image),width,height))
                finally:
                    bpy.data.images.remove(image)
            v = self.space.region_3d
            self.offscreen.draw_view3d(self.scene, context.view_layer, self.space, self.region,
                v.view_matrix, v.window_matrix, do_color_management=False)
            self.frames.append(copy_texture(self.offscreen.texture_color,self.width,self.height))
            self.scene.flowhmr.message = f'Preparing playback · {self.next_frame-self.scene.frame_start+1} / {self.scene.frame_end-self.scene.frame_start+1} frames · Esc cancels'
            self.area.header_text_set(self.scene.flowhmr.message)
            self.area.tag_redraw()
            self.next_frame += 1
            return {'RUNNING_MODAL'}
        except Exception as error:
            import traceback; traceback.print_exc()
            self.finish(context,False)
            self.report({'ERROR'},str(error))
            return {'CANCELLED'}


@persistent
def invalidate(scene, depsgraph):
    if _CACHE is None or _BUILD is not None:return
    from . import ui
    if ui._PLAYBACK is not None:return
    if scene.frame_current != _CACHE['last_frame']:
        _CACHE['last_frame']=scene.frame_current
        return
    if any(u.is_updated_geometry or u.is_updated_transform or isinstance(u.id,bpy.types.Material) for u in depsgraph.updates):
        clear()
