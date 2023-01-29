from .resnet_fpn import ResNetFPN_8_2, ResNetFPN_16_4, ResNetFPN_8_2_P
from .davit import DaViTFPN
from .max_vit import MaxViTFPN

def build_backbone(config):
    if config['backbone_type'] == 'ResNetFPN':
        if config['resolution'] == (8, 2):
            return ResNetFPN_8_2_P(config['resnetfpn'])
        elif config['resolution'] == (16, 4):
            return ResNetFPN_16_4(config['resnetfpn'])
    elif config['backbone_type'] == 'DaViTFPN':
        return DaViTFPN(config['davitfpn'])
    elif config['backbone_type'] == 'MaxViTFPN':
        return MaxViTFPN(config['maxvitfpn'])
    else:
        raise ValueError(f"LOFTR.BACKBONE_TYPE {config['backbone_type']} not supported.")
