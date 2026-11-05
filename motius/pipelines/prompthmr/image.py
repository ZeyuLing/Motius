"""Neural image graph and separate licensed geometry for prepared image batches."""
import numpy as np
import torch


class PromptHMRImagePrediction:
    def __init__(self,bundle,geometry):
        self.bundle=bundle
        self.geometry=geometry.to(bundle.device)

    def __call__(self,batch,*,box_prompt=True,kpt_prompt=True,text_prompt=True,mask_prompt=True,interaction_prompt=True,use_mean_hands=False):
        bundle=self.bundle;device=bundle.device
        intrinsics=torch.cat([item['cam_int'] for item in batch]).to(device)
        images=torch.stack([item['image'] for item in batch]).to(device)
        embeddings=bundle.encode_images(images,intrinsics)
        outputs=[]
        for index,item in enumerate(batch):
            text=item.get('text') if text_prompt else None
            null_mask=None
            if text is not None:
                import open_clip
                null_mask=torch.tensor(np.array(text)=='NULL',device=device)
                text=bundle.encode_text_tokens(open_clip.tokenize(text).to(device))
            sparse,dense=bundle.encode_prompts(item.get('boxes') if box_prompt else None,text,
                item.get('kpts') if kpt_prompt else None,item.get('masks') if mask_prompt else None,text_null_mask=null_mask)
            prompt=getattr(bundle.prompt_encoder,'module',bundle.prompt_encoder)
            pe=prompt.get_dense_pe()
            intrinsic=item['cam_int'].to(device)
            prediction=bundle.decode_image(intrinsic,embeddings[[index]],pe,sparse,dense,
                item.get('interaction',False) if interaction_prompt else False)
            output=dict(pose=prediction[0],betas=prediction[1],transl=prediction[2],transl_c=prediction[3],
                inv_depth_c=prediction[4],cam_int=intrinsic.expand(len(sparse),3,3),features=prediction[5])
            outputs.append(self.geometry.process_output(output,use_mean_hands))
        return outputs
