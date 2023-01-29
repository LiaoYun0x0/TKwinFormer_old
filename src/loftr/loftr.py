import torch
import torch.nn as nn
from einops.einops import rearrange
import matplotlib.pyplot as plt

from .backbone import build_backbone
from .utils.position_encoding import PositionEncodingSine
from .loftr_module import LocalFeatureTransformer, FinePreprocess
from .utils.coarse_matching import CoarseMatching
from .utils.fine_matching import FineMatching
from .loftr_module.transformer import Mlp


class LoFTR(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config

        # Modules
        self.backbone = build_backbone(config)
        self.pos_encoding = PositionEncodingSine(
            config['coarse']['d_model'],
            temp_bug_fix=config['coarse']['temp_bug_fix'])
        self.loftr_coarse = LocalFeatureTransformer(config['coarse'])
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.fine_preprocess = FinePreprocess(config)
        self.loftr_fine = LocalFeatureTransformer(config["fine"])
        self.fine_matching = FineMatching()
        
        # self.mlp = Mlp(config['coarse']['d_model'])
    def forward(self, data):
        """ 
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        # 1. Local Feature CNN
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })

        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            feats_c, feats_f = self.backbone(torch.cat([data['image0'], data['image1']], dim=0))
            import pdb
            pdb.set_trace()
            (feat_c0, feat_c1), (feat_f0, feat_f1) = feats_c.split(data['bs']), feats_f.split(data['bs'])
        else:  # handle different input shapes
            (feat_c0, feat_f0), (feat_c1, feat_f1) = self.backbone(data['image0']), self.backbone(data['image1'])
        """
        plt.imshow(feat_c0[0, 0, :, :], cmap=plt.cm.jet)
        plt.savefig("t1.png")
        plt.imshow(feat_c1[0, 0, :, :], cmap=plt.cm.jet)
        plt.savefig("t2.png")
        n, c, h, w = feat_c0.shape
        """

        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
            'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]
        })

        # 2. coarse-level loftr module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        # feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
        # feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')
        feat_c0 = self.pos_encoding(feat_c0)
        feat_c1 = self.pos_encoding(feat_c1)
        # b,c,h,w = feat_c0.shape
        # feat_c0 = self.mlp(rearrange(feat_c0,'b c h w -> b (h w) c'),h,w).transpose(1,2).reshape(b,c,h,w)
        # feat_c1 = self.mlp(rearrange(feat_c1,'b c h w -> b (h w) c'),h,w).transpose(1,2).reshape(b,c,h,w)

        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)
        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)
        if len(feat_c0.shape) == 4:
            feat_c0 = rearrange(feat_c0, 'b c h w -> b (h w) c')
            feat_c1 = rearrange(feat_c1, 'b c h w -> b (h w) c')
        """
        feat_c0_0 = rearrange(feat_c0, 'n (h w) c-> n c h w', h=h, w=w)
        feat_c1_0 = rearrange(feat_c1, 'n (h w) c-> n c h w', h=h,w=w)
        plt.imshow(feat_c0_0[0, 0, :, :], cmap=plt.cm.jet)
        plt.savefig("t3.png")
        plt.imshow(feat_c1_0[0, 0, :, :], cmap=plt.cm.jet)
        plt.savefig("t4.png")
        """


        # 3. match coarse-level
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)

        # 4. fine-level refinement
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_f0, feat_f1, feat_c0, feat_c1, data)
        if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
            feat_f0_unfold, feat_f1_unfold = self.loftr_fine(feat_f0_unfold, feat_f1_unfold)

        # 5. match fine-level
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
