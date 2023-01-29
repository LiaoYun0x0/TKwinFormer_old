import os
from copy import deepcopy

import torch
import cv2
import numpy as np
import matplotlib.cm as cm
from src.utils.plotting import make_matching_figure
import random

from src.loftr import LoFTR
from src.config.default import default_cfg

# The default config uses dual-softmax.
# The outdoor and indoor models share the same config.
# You can change the default values like thr and coarse_match_type.
_default_cfg = deepcopy(default_cfg)
_default_cfg['loftr']['coarse']['temp_bug_fix'] = True  # set to False when using the old ckpt
matcher = LoFTR(config=_default_cfg['loftr'])
#matcher.load_state_dict(torch.load("/first_disk/TopKWindows/LoFTR_DaViT/logs/tb_logs/TopKWindowLinearAttentionV2_10%_normPE/version_19/checkpoints/epoch=12-auc@5=0.515-auc@10=0.681-auc@20=0.803.ckpt")['state_dict'])
#matcher = matcher.eval().cuda()
matcher = matcher.eval().cuda()
device='cuda'

def frame2tensor(frame, device):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return torch.from_numpy(frame/255.).float()[None, None].to(device)

image0_paths_ = ["/first_disk/TopKWindows/LoFTR_DaViT/data/megadepth/test/phoenix/S6/zl548/MegaDepth_v1/0022/dense0/imgs/12003890_f6c899bec0_o.jpg"]
image1_paths_ = ["/first_disk/TopKWindows/LoFTR_DaViT/data/megadepth/test/phoenix/S6/zl548/MegaDepth_v1/0022/dense0/imgs/19790727_79034989f3_o.jpg"]
for image0_path, image1_path in zip(image0_paths_, image1_paths_):
    image0 = cv2.imread(image0_path)
    image1 = cv2.imread(image1_path)

    image0_h, image0_w, c = image0.shape
    image1_h, image1_w, c = image1.shape

    scale0 = (896 / image0_w, 896 / image0_h)
    scale1 = (896 / image1_w, 896 / image1_h)


    image0 = cv2.resize(image0, (896, 896))
    image1 = cv2.resize(image1, (896, 896))

    frame_tensor = frame2tensor(image0, device)
    last_data = {'image0': frame_tensor, 'src_image0': image0}

    frame_tensor = frame2tensor(image1, device)
    batch = {**last_data, 'image1': frame_tensor, 'src_image1': image1}


    # Inference with LoFTR and get prediction
    with torch.no_grad():
        matcher(batch)
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        mconf = batch['mconf'].cpu().numpy()

        src_mkpts0 = mkpts0 / scale0
        src_mkpts1 = mkpts1 / scale1

        out_img = np.concatenate([image0,image1],axis=1).copy()
        h,w, c = image1.shape
        H,mask = cv2.findHomography(mkpts0, mkpts1,cv2.RANSAC,ransacReprojThreshold=16)

        for i in range(mkpts0.shape[0]):
            cv2.line(out_img,(int(mkpts0[i,0]),int(mkpts0[i,1])),(int(mkpts1[i,0])+w,int(mkpts1[i,1])),(0,255,0),1)

        out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"match_result/{os.path.basename(image0_path)}-{os.path.basename(image1_path)}.jpg", out_img)

