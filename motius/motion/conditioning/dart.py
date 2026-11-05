import torch
import clip

def have_overlap(seg1, seg2):
    if seg1[0] > seg2[1] or seg2[0] > seg1[1]:
        return False
    else:
        return True

def get_overlap(seg1, seg2):
    overlap_len = max(0, min(seg1[1], seg2[1]) - max(seg1[0], seg2[0]))
    return overlap_len

def encode_text(clip_model, raw_text, force_empty_zero=True):
    device = next(clip_model.parameters()).device
    texts = clip.tokenize(raw_text, truncate=True).to(device)
    text_embedding = clip_model.encode_text(texts).float()
    if force_empty_zero:
        empty_text = [text == '' for text in raw_text]
        text_embedding[empty_text, :] = 0
    return text_embedding

def compose_texts_with_and(texts):
    texts = sorted(texts)
    return ' and '.join(texts)
