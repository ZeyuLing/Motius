"""Full-frame diagonal focal conversion."""
import numpy as np


def focal_length_from_mm(width, height, mm=24):
    """
    Convert full-frame focal length to image sensor focal length.
    """
    diag_fullframe = (24 ** 2 + 36 ** 2) ** 0.5
    diag_img = (width ** 2 + height ** 2) ** 0.5
    focal_length = diag_img / diag_fullframe * mm
    return focal_length
