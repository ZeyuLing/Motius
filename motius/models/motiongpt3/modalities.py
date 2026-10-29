import torch
from torch import nn

class ModalityInfo:

    def __init__(self, modality_name: str, modality_type: int, som_id: int, eom_id: int, pad_id: int, msk_id: int=None, token_id_start: int=0, pre_processor: nn.Module=nn.Identity(), post_processor: nn.Module=nn.Identity(), som_generated: bool=False, eom_generated: bool=False, loss_fct: nn.Module=None, mod_voc_size: int=None):
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

def get_modalities_infos(config, tokenizer) -> dict:
    info_list = []
    token_id_start = 0
    text_loss_fct = None
    text_modality_info = ModalityInfo(modality_type=1, modality_name='text', token_id_start=token_id_start, mod_voc_size=config.text_vocab_size, som_id=tokenizer.bos_token_id, eom_id=tokenizer.eos_token_id, pad_id=tokenizer.pad_token_id, pre_processor=None, post_processor=None, loss_fct=text_loss_fct)
    info_list.append(text_modality_info)
    token_id_start += config.text_vocab_size
    som_id, eom_id = tokenizer.convert_tokens_to_ids(['<start_of_motion>', '<end_of_motion>'])
    msk_id, pad_id = tokenizer.convert_tokens_to_ids(['<masked_motion>', '<pad_motion>'])
    motion_und_head = nn.Embedding(config.motion_vocab_size, config.mot_embed_dim, padding_idx=pad_id - token_id_start)
    motion_gen_head = nn.Linear(config.mot_embed_dim, config.motion_vocab_size, bias=False)
    torch.nn.init.normal_(motion_gen_head.weight, mean=0.0, std=0.01)
    mot_loss_fct = None
    motion_modality_info = ModalityInfo(modality_type=2, modality_name='motion', mod_voc_size=config.motion_vocab_size, token_id_start=token_id_start, pad_id=pad_id - token_id_start, som_id=som_id - token_id_start, eom_id=eom_id - token_id_start, msk_id=msk_id - token_id_start, pre_processor=motion_und_head, post_processor=motion_gen_head, loss_fct=mot_loss_fct)
    info_list.append(motion_modality_info)
    token_id_start += config.motion_vocab_size
    return info_list
