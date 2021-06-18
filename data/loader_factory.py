from data.voc_loader_erase import VOCErase
from data.voc_loader_erasebb import VOCEraseBB

# VOC ***************************************************************************
def get_voc_erase_base(source='train'):
    return VOCErase(source, erase_size=0)

def get_voc_erase_gaus(source='train', blur_size=0):
    return VOCErase(source, erase_size=blur_size)

def get_voc_erase_bbox(source='train'):
    return VOCEraseBB()

def get_voc_mask_base(source='train'):
    pass

def get_voc_mask_bbbx(source='train'):
    pass

# VOCO ***************************************************************************
def get_voco_erase(source='train'):
    pass

def get_voco_mask(source='train'):
    pass
