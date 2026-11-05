# SPDX-License-Identifier: GPL-3.0-or-later
"""Native viewport subject picker. Image and hit tests share one transform."""
import json
from pathlib import Path
import bpy
import blf
import gpu
from gpu_extras.batch import batch_for_shader
from bpy.types import Operator
from .subjects import candidates, hit_test, digest, load_subjects

INK = (0.94, 0.96, 0.97, 1.)
MUTED = (0.65, 0.71, 0.75, 1.)
ACCENT = (0.22, 0.83, 0.69, 1.)


def text(x, y, value, size=18, color=INK):
    blf.position(0, x, y, 0); blf.size(0, size); blf.color(0, *color); blf.draw(0, value)


def rectangle(x, y, w, h, color, outline=False):
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    points = [(x,y),(x+w,y),(x+w,y+h),(x,y+h)]
    batch = batch_for_shader(shader, 'LINE_LOOP' if outline else 'TRI_FAN', {'pos': points})
    shader.bind(); shader.uniform_float('color', color); batch.draw(shader)


class FLOWHMR_OT_choose_subject(Operator):
    bl_idname = 'flowhmr.choose_subject'
    bl_label = 'Choose Person'
    bl_description = 'Click a person, Tab cycles overlapping boxes, arrows review the track, Enter confirms'

    def load_frame(self):
        if self.image:
            bpy.data.images.remove(self.image)
        self.frame = self.frames[self.index]
        self.image = bpy.data.images.load(str(self.root/'previews'/f'{self.frame:06d}.jpg'), check_existing=False)
        self.texture = gpu.texture.from_image(self.image)
        self.hits = []

    def transform(self):
        w, h = self.region.width, self.region.height
        scale = min((w-80)/self.data['width'], (h-190)/self.data['height'])
        return (w-self.data['width']*scale)/2, 100, scale

    def draw_canvas(self):
        if bpy.context.area != self.area:
            return
        w, h = self.region.width, self.region.height
        gpu.state.blend_set('ALPHA')
        rectangle(0, 0, w, h, (.055,.071,.082,1))
        text(40, h-46, 'Choose who moves.', 28)
        text(40, h-77, 'Click a person. Review their track before generating motion.', 17, MUTED)
        x,y,s = self.transform(); iw, ih = self.data['width']*s, self.data['height']*s
        shader = gpu.shader.from_builtin('IMAGE')
        batch = batch_for_shader(shader, 'TRI_FAN', {'pos':[(x,y),(x+iw,y),(x+iw,y+ih),(x,y+ih)], 'texCoord':[(0,0),(1,0),(1,1),(0,1)]})
        shader.bind(); shader.uniform_sampler('image',self.texture);batch.draw(shader)
        gpu.state.line_width_set(3.)
        for tid, box in candidates(self.data, self.frame):
            a,b,c,d = box; color = ACCENT if tid == self.selected else INK
            rectangle(x+a*s, y+(self.data['height']-d)*s, (c-a)*s, (d-b)*s, color, True)
            label = f'Person {tid}' + (' · Selected' if tid == self.selected else '')
            tx,ty = x+a*s,max(y+8,y+(self.data['height']-b)*s-25)
            rectangle(tx,ty-4,len(label)*9+16,27,(.035,.05,.06,.92));text(tx+7,ty+2,label,16,color)
        gpu.state.line_width_set(1.)
        present = self.selected in [tid for tid,_ in candidates(self.data,self.frame)]
        title = f'Person {self.selected} selected' if self.selected >= 0 else 'No person selected'
        if self.selected >= 0 and not present: title += ' · not detected at this frame'
        if len(self.hits)>1: title += f' · {len(self.hits)} overlapping boxes: Tab to cycle'
        text(40,68,title,18,ACCENT if present else INK)
        text(40,38,f'Frame {self.frame+1} / {self.data["frames"]}    ← → / wheel: review    Tab: next overlap    Esc: cancel',15,MUTED)
        rectangle(w-230,h-72,190,40,ACCENT if self.selected>=0 else (.18,.22,.25,1))
        text(w-213,h-58,'Confirm person ↵',17,(.035,.05,.06,1) if self.selected>=0 else MUTED)
        gpu.state.blend_set('NONE')

    def finish(self, context):
        bpy.types.SpaceView3D.draw_handler_remove(self.handle,'WINDOW')
        if self.image: bpy.data.images.remove(self.image)
        self.area.spaces.active.show_region_ui = self.saved_sidebar
        self.area.spaces.active.show_region_tool_header = self.saved_tool_header
        context.window.cursor_modal_restore();self.area.tag_redraw()

    def invoke(self, context, event):
        try:
            props = context.scene.flowhmr
            self.root = Path(bpy.path.abspath(props.subjects_file)).parent
            self.data = load_subjects(bpy.path.abspath(props.subjects_file))
            if self.data['video_sha256'] != digest(bpy.path.abspath(props.video)):
                raise ValueError('Video changed. Detect people again before choosing a subject.')
            self.frames = sorted(int(p.stem) for p in (self.root/'previews').glob('*.jpg') if int(p.stem)%30==0 or int(p.stem)==self.data['frames']-1)
            if not self.frames or not self.data['tracks']: raise ValueError('No people found. Try a clearer visible segment.')
            self.area, self.region = context.area, next(r for r in context.area.regions if r.type=='WINDOW')
            self.saved_sidebar=context.space_data.show_region_ui
            self.saved_tool_header=context.space_data.show_region_tool_header
            context.space_data.show_region_ui=False
            context.space_data.show_region_tool_header=False
            self.index=0;self.image=None;self.selected=props.subject_id;self.hits=[]
            self.load_frame()
            self.handle=bpy.types.SpaceView3D.draw_handler_add(self.draw_canvas,(),'WINDOW','POST_PIXEL')
            context.window.cursor_modal_set('CROSSHAIR');context.window_manager.modal_handler_add(self)
            self.area.tag_redraw();return {'RUNNING_MODAL'}
        except Exception as error:
            self.report({'ERROR'},str(error));return {'CANCELLED'}

    def modal(self,context,event):
        if event.type=='ESC' and event.value=='PRESS':
            self.finish(context);return {'CANCELLED'}
        if event.value!='PRESS': return {'RUNNING_MODAL'}
        mx,my = event.mouse_x-self.region.x,event.mouse_y-self.region.y
        confirm = event.type in {'RET','NUMPAD_ENTER'} or (event.type=='LEFTMOUSE' and mx>self.region.width-230 and my>self.region.height-72)
        if confirm and self.selected>=0:
            from .subject_overlay import install_markers
            try:
                install_markers(context, self.data, self.selected)
            except Exception as error:
                self.finish(context)
                self.report({'ERROR'}, str(error))
                return {'CANCELLED'}
            context.scene.flowhmr.subject_id=self.selected
            context.scene.flowhmr.message=f'Person {self.selected} selected · review complete'
            self.finish(context);return {'FINISHED'}
        if event.type in {'RIGHT_ARROW','WHEELDOWNMOUSE','LEFT_ARROW','WHEELUPMOUSE'}:
            self.index=max(0,min(len(self.frames)-1,self.index+(1 if event.type in {'RIGHT_ARROW','WHEELDOWNMOUSE'} else -1)))
            self.load_frame()
        elif event.type=='TAB' and self.hits:
            self.selected=self.hits[(self.hits.index(self.selected)+1)%len(self.hits)]
        elif event.type=='LEFTMOUSE':
            x,y,s=self.transform();point=((mx-x)/s,self.data['height']-(my-y)/s)
            hits=hit_test(candidates(self.data,self.frame),point)
            if hits:
                self.selected=hits[(hits.index(self.selected)+1)%len(hits)] if hits==self.hits and self.selected in hits else hits[0]
            self.hits=hits
        self.area.tag_redraw();return {'RUNNING_MODAL'}
