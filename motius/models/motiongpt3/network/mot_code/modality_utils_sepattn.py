import torch
from torch import nn

# from typing import NamedTuple
class ModalityInfo:
    def __init__(self,
                 modality_name: str,
                 modality_type: int,
                 som_id: int,
                 eom_id: int,
                 pad_id: int,
                 msk_id: int = None,
                 token_id_start: int = 0,
                 pre_processor: nn.Module = nn.Identity(),
                 post_processor: nn.Module = nn.Identity(),
                 som_generated: bool = False,
                 eom_generated: bool = False,
                 loss_fct: nn.Module = None,
                 mod_voc_size: int = None
                ):
        self.modality_name = modality_name
        self.modality_type = modality_type
        self.som_id = som_id
        self.eom_id = eom_id
        self.msk_id = msk_id
        self.pad_id = pad_id
        self.token_id_start = token_id_start
        self.label_processor = nn.Identity()
        self.pre_processor = pre_processor
        self.post_processor = post_processor
        self.som_generated = som_generated
        self.eom_generated = eom_generated
        self.loss_fct = loss_fct
        self.mod_voc_size = mod_voc_size
    

class MotionUndHead(nn.Module):
    def __init__(self, input_dim, output_dim, projector_type='linear', depth=1, **kwargs):
        super().__init__()
        self.output_dim = output_dim
        if projector_type == "identity":
            modules = nn.Identity()

        elif projector_type == "linear":
            modules = nn.Linear(input_dim, output_dim)

        elif projector_type == "mlp_gelu":
            mlp_depth = depth
            modules = [nn.Linear(input_dim, output_dim)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(output_dim, output_dim))
            modules = nn.Sequential(*modules)
        
        else:
            raise ValueError(f"Unknown projector type: {projector_type}")
        
        self.layers = modules

    def forward(self, motion_tokens):
        motion_embedding = self.layers(motion_tokens)
        return motion_embedding

class LossSkip(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
    def forward(sellf, inputs, targets):
        return torch.tensor(0.).to(inputs)

class MoTLoss(nn.Module):
    def __init__(self, loss_fct):
        super().__init__()
        self.loss_fct = loss_fct
    
    def forward(self, inputs, target, valid_pos):
        if valid_pos.sum() == 0: return torch.tensor(0.).to(inputs)
        inputs = inputs[valid_pos].contiguous()
        target = target[valid_pos].contiguous()
        loss_mod = self.loss_fct(inputs.view(-1, inputs.size(-1)), target.view(-1))
        return loss_mod

class MoTShiftCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super().__init__()
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(self, inputs, target, valid_pos):
        valid_pos = valid_pos[:,1:]
        if valid_pos.sum() == 0: return torch.tensor(0.).to(inputs)
        shift_logits = inputs[..., :-1, :][valid_pos].contiguous()
        shift_labels = target[..., 1:][valid_pos].contiguous()
        loss_mod = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss_mod

class MoTCrossEntropyLoss(MoTLoss):
    def __init__(self, ignore_index=-100):
        super().__init__()
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        
class MotL2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fct = nn.MSELoss()

    def forward(self, inputs, target, valid_pos):
        if isinstance(target, list):
            target = torch.cat(target, 0)
        inputs = inputs[valid_pos].contiguous()
        # target = target[valid_pos].contiguous()
        loss_mod = self.loss_fct(inputs, target.to(inputs.device))
        return loss_mod
    
def get_modalities_infos(config, tokenizer)-> dict:
    info_list = []
    token_id_start = 0
    # shared = nn.Embedding(config.vocab_size, config.d_model, config.pad_token_id)
    # lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
    text_loss_fct = MoTCrossEntropyLoss(ignore_index=-100) if config.is_encoder_decoder else MoTShiftCrossEntropyLoss(ignore_index=-100)
    text_modality_info = ModalityInfo(
        modality_type=1,
        modality_name='text',
        token_id_start = token_id_start,
        mod_voc_size=config.text_vocab_size,
        som_id=tokenizer.bos_token_id, 
        eom_id=tokenizer.eos_token_id,
        pad_id=tokenizer.pad_token_id,
        pre_processor=None, #shared,
        post_processor=None, #lm_head,
        loss_fct=text_loss_fct)
    # info_list['text'] = text_modality_info
    info_list.append(text_modality_info)
    token_id_start += config.text_vocab_size

    som_id, eom_id =  tokenizer.convert_tokens_to_ids(['<start_of_motion>', '<end_of_motion>'])
    msk_id, pad_id = tokenizer.convert_tokens_to_ids(['<masked_motion>', '<pad_motion>'])

    motion_und_head = nn.Embedding(config.motion_vocab_size, config.mot_embed_dim, padding_idx=pad_id-token_id_start)
    motion_gen_head = nn.Linear(config.mot_embed_dim, config.motion_vocab_size, bias=False)
    # torch.nn.init.normal_(motion_und_head, std=.02)
    torch.nn.init.normal_(motion_gen_head.weight, mean=0.0, std=.01)

    if config.mot_loss == 'ce':
        mot_loss_fct = MoTCrossEntropyLoss(ignore_index=-100) if config.is_encoder_decoder else MoTShiftCrossEntropyLoss(ignore_index=-100)
    elif config.mot_loss in ['l2', 'skip']:
        mot_loss_fct=MoTLoss(LossSkip())
    else:
        assert False, f'unknown mot_loss, {config.mot_loss}'

    motion_modality_info = ModalityInfo(
        modality_type=2,
        modality_name='motion',
        mod_voc_size=config.motion_vocab_size,
        token_id_start=token_id_start,
        pad_id=pad_id - token_id_start,
        som_id=som_id - token_id_start, 
        eom_id=eom_id - token_id_start,
        msk_id=msk_id - token_id_start,
        pre_processor=motion_und_head,
        post_processor=motion_gen_head,
        loss_fct=mot_loss_fct,
        # loss_fct=torch.nn.CrossEntropyLoss(ignore_index=-100)
        )
    info_list.append(motion_modality_info)
    token_id_start += config.motion_vocab_size
    return info_list
