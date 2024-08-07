import argparse
import os
import os.path as osp
import sys

import deepdish as dd
import numpy as np
import torch
import torchvision.transforms as transforms

SMPLERX_ROOT = '/mnt/vol_c/projects/synth-smplerx'
sys.path.insert(0, osp.join(SMPLERX_ROOT, 'main'))
sys.path.insert(0, osp.join(SMPLERX_ROOT, 'data'))
sys.path.insert(0, osp.join(SMPLERX_ROOT, 'common'))
sys.path.insert(0, osp.join(SMPLERX_ROOT))

from config import cfg
from tqdm import tqdm
from typing import Optional
from mmdet.apis import init_detector, inference_detector
from utils.inference_utils import process_mmdet_results

config_path = f'{SMPLERX_ROOT}/main/config/config_smpler_x_s_synth.py'
ckpt_path = f'{SMPLERX_ROOT}/smplerx_1_5M/snapshot_9.pth.tar'
# ckpt_path = '../../synth_small_20240415_121041/model_dump/snapshot_9.pth.tar'

cfg.get_config_fromfile(config_path)
cfg.update_test_config('EHF', 'na', shapy_eval_split=None, pretrained_model_path=ckpt_path, use_cache=False)
cfg.update_config(1, 'output/test')
cfg.encoder_config_file = f'{SMPLERX_ROOT}/main/transformer_utils/configs/smpler_x/encoder/body_encoder_small.py'

# load model
from base import Demoer
from utils.preprocessing import process_bbox, generate_patch_image, generate_patch_image_preserve_aspect_ratio
from eval.egobody import EgoBodyDataset
from eval.ehf import EHFDataset
from eval.ssp3d import SSP3DDataset
from eval.ubody import UBodyDataset


def make_pred(image, bbox_xywh: Optional[np.array] = None, bbox_scale: float = 1.2):
    original_img_height, original_img_width = image.shape[:2]
    transform = transforms.ToTensor()

    if bbox_xywh is None:
        ## mmdet inference
        mmdet_results = inference_detector(model, image)
        mmdet_box = process_mmdet_results(mmdet_results, cat_id=0, multi_person=True)[0]

        bbox_id = 0
        mmdet_box_xywh = np.zeros((4))
        mmdet_box_xywh[0] = mmdet_box[bbox_id][0]
        mmdet_box_xywh[1] = mmdet_box[bbox_id][1]
        mmdet_box_xywh[2] = abs(mmdet_box[bbox_id][2] - mmdet_box[bbox_id][0])
        mmdet_box_xywh[3] = abs(mmdet_box[bbox_id][3] - mmdet_box[bbox_id][1])
    else:
        mmdet_box_xywh = bbox_xywh

    # bbox = process_bbox(mmdet_box_xywh, original_img_width, original_img_height, bbox_scale)
    bbox = mmdet_box_xywh
    bbox = process_bbox(bbox, original_img_width, original_img_height, ratio=bbox_scale)
    img, *_ = generate_patch_image(image, bbox, 1.0, 0.0, False, cfg.input_img_shape)

    img = transform(img.astype(np.float32)) / 255.
    img = img.cuda()[None, :, :, :]
    inputs = {'img': img}
    targets = {}
    meta_info = {}

    with torch.no_grad():
        out = demoer.model(inputs, targets, meta_info, 'test')
    # mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]

    smplx_pred = {}
    smplx_pred['global_orient'] = out['smplx_root_pose'].reshape(-1, 3).cpu().numpy()
    smplx_pred['body_pose'] = out['smplx_body_pose'].reshape(-1, 3).cpu().numpy()
    smplx_pred['left_hand_pose'] = out['smplx_lhand_pose'].reshape(-1, 3).cpu().numpy()
    smplx_pred['right_hand_pose'] = out['smplx_rhand_pose'].reshape(-1, 3).cpu().numpy()
    smplx_pred['jaw_pose'] = out['smplx_jaw_pose'].reshape(-1, 3).cpu().numpy()
    smplx_pred['leye_pose'] = np.zeros((1, 3), dtype=np.float32)
    smplx_pred['reye_pose'] = np.zeros((1, 3), dtype=np.float32)
    smplx_pred['betas'] = out['smplx_shape'].reshape(-1, 10).cpu().numpy()[0]
    smplx_pred['expression'] = out['smplx_expr'].reshape(-1, 10).cpu().numpy()[0]
    smplx_pred['transl'] = out['cam_trans'].reshape(-1, 3).cpu().numpy()[0]

    focal = [cfg.focal[0] / cfg.input_body_shape[1] * bbox[2], cfg.focal[1] / cfg.input_body_shape[0] * bbox[3]]
    princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0],
               cfg.princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]]

    smplx_pred['focal'] = np.array(focal, dtype=np.float32)
    smplx_pred['princpt'] = np.array(princpt, dtype=np.float32)
    smplx_pred['bbox'] = mmdet_box_xywh.astype(np.float32)
    # smplx_pred["vertices"] = mesh
    # smplx_pred["faces"] = demoer.model.module.smplx_layer.faces.astype(np.int64)
    return smplx_pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process a folder.")
    parser.add_argument("--dataset", choices=["ehf", "egobody", "ssp3d", "ubody"])
    parser.add_argument("--experiment_tag", type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--bbox_scale", default=1.1, type=float)


    args = parser.parse_args()

    if args.dataset == "ehf":
        dataset = EHFDataset(
            data_path='/mnt/vol_d/datasets/EHF',
            smplx_model_path='/mnt/vol_c/projects/synth-smplerx/common/utils/human_model_files/smplx/',
        )
    elif args.dataset == "egobody":
        dataset = EgoBodyDataset(
            data_path='/mnt/vol_d/datasets/EgoBody',
            smpl_model_path='/mnt/vol_c/projects/synth-smplerx/common/utils/human_model_files/smpl/',
        )
    elif args.dataset == "ssp3d":
        dataset = SSP3DDataset(
            data_path='/mnt/vol_f/smplify_cse/ssp3d/',
            smpl_model_path='/mnt/vol_c/projects/synth-smplerx/common/utils/human_model_files/smpl/',
        )
    elif args.dataset == "ubody":
        dataset = UBodyDataset(
            data_path="/mnt/vol_d/datasets",
            smplx_model_path="/mnt/vol_c/projects/synth-smplerx/common/utils/human_model_files/smplx/",
        )
    else:
        raise ValueError

    device = args.device

    output_path = f"prediction/{args.dataset}/{args.experiment_tag}"
    os.makedirs(output_path, exist_ok=True)

    demoer = Demoer()
    demoer._make_model()
    _ = demoer.model.eval()

    checkpoint_file = f'{SMPLERX_ROOT}/pretrained_models/mmdet/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    config_file = f'{SMPLERX_ROOT}/pretrained_models/mmdet/mmdet_faster_rcnn_r50_fpn_coco.py'
    model = init_detector(config_file, checkpoint_file, device=device)

    for i in tqdm(range(len(dataset))):
        instance = dataset[i]
        if dataset.HAS_GT_BBOX:
            image, bbox, *_ = instance
        else:
            image, *_ = instance
            bbox = None
        output = make_pred(image, bbox, bbox_scale=args.bbox_scale)

        save_path = os.path.join(output_path, f'{i}.h5')
        dd.io.save(save_path, output)
