"""Optional camera and ground-plane previews using native materialized meshes."""
from pathlib import Path
import numpy as np
import torch
from smplx import SMPLLayer
from motius.visualization.gvhmr.renderer import Renderer,get_global_cameras_static,get_ground_params_from_points
from motius.motion.representation.gvhmr.camera import create_camera_sensor
from motius.motion.representation.gvhmr.geometry import compute_T_ayfz2ay,apply_T_on_points
from .video import get_video_lwh,get_video_reader,get_writer
from .materialization import materialize_prediction


@torch.no_grad()
def render_prediction(prediction,video,output,body_model_path,device,*,faces=None):
    if "vertices_camera" not in prediction:
        prediction=materialize_prediction(prediction,body_model_path,device)
    if faces is None:
        body_root=Path(body_model_path).parent.parent
        faces=SMPLLayer(model_path=str(body_root/"smpl/SMPL_NEUTRAL.pkl")).faces
    frames,width,height=get_video_lwh(str(video))
    camera=prediction["vertices_camera"].to(device)
    renderer=Renderer(width,height,device=device,faces=faces,K=prediction["K_fullimg"][0].to(device))
    reader=get_video_reader(str(video));writer=get_writer(str(Path(output)/"1_incam.mp4"),fps=30,crf=23)
    try:
        for i,image in enumerate(reader):writer.write_frame(renderer.render_mesh(camera[i],image,[.8,.8,.8]))
    finally:writer.close();reader.close()
    vertices=prediction["vertices_world"].to(device).clone()
    joints=prediction["joints_world"].to(device)
    offset=joints[0,0].clone();offset[1]=vertices[:,:,[1]].min();vertices=vertices-offset
    joints=joints-offset
    transform=compute_T_ayfz2ay(joints[[0]],inverse=True)
    vertices=apply_T_on_points(vertices,transform)
    joints=apply_T_on_points(joints,transform)
    rotation,translation,lights=get_global_cameras_static(vertices.cpu(),beta=2.,cam_height_degree=20,target_center_height=1.,device=device)
    K=create_camera_sensor(width,height,24)[2].to(device)
    renderer=Renderer(width,height,device=device,faces=faces,K=K)
    scale,cx,cz=get_ground_params_from_points(joints[:,0],vertices)
    renderer.set_ground(scale*1.5,cx,cz)
    writer=get_writer(str(Path(output)/"2_global.mp4"),fps=30,crf=23)
    try:
        for i in range(frames):
            cameras=renderer.create_camera(rotation[i],translation[i])
            image=renderer.render_with_ground(vertices[[i]],torch.ones(1,3,device=device)*.8,cameras,lights)
            writer.write_frame(image)
    finally:writer.close()
