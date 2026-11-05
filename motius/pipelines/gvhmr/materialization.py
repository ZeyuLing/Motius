"""Full released SMPL-X body geometry and sparse SMPL24 materialization."""
from pathlib import Path
import torch
from motius.motion.body_models.gvhmr.body import BodyModelSMPLX


@torch.no_grad()
def materialize_prediction(prediction,body_model_path,device):
    model=BodyModelSMPLX(model_path=str(body_model_path),model_type="smplx",gender="neutral",
        num_pca_comps=12,flat_hand_mean=False).to(device).eval()
    asset_root=Path(__file__).resolve().parents[2]/"configuration/gvhmr/body"
    mapping=torch.load(asset_root/"smplx2smpl_sparse.pt",map_location=device,weights_only=True)
    regressor=torch.load(asset_root/"smpl_neutral_J_regressor.pt",map_location=device,weights_only=True)
    output=dict(prediction)
    for name,group in [("camera","smpl_params_incam"),("world","smpl_params_global")]:
        vertices=model(**{k:v.to(device) for k,v in prediction[group].items()}).vertices
        vertices=torch.stack([torch.matmul(mapping,frame) for frame in vertices])
        output[f"vertices_{name}"]=vertices.cpu()
        output[f"joints_{name}"]=torch.einsum("jv,fvc->fjc",regressor,vertices).cpu()
    return output
