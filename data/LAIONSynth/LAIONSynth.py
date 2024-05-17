import glob
import os

import cv2
import deepdish as dd
import numpy as np
import torch
from config import cfg
from torch.utils.data import Dataset
from utils.human_models import smpl_x
from utils.preprocessing import augmentation

markers_idxs = [
    7780, 5032, 8043, 5292, 8080, 5334, 7497, 4761,  # 0 - 7
    7107, 4371, 6996, 4252, 7210, 4474, 6323, 3562,  # 8 - 15
    6150, 3389, 8320, 5626, 7142, 4406, 6651, 3903,  # 16 - 23
    8527, 5780, 6443, 3682, 6050, 3287, 6265, 3504,  # 24 - 31
    7176, 5465, 6844, 4100, 7129, 4393, 8339, 5646,  # 32 - 39
    8635, 8847, 8245, 5525, 6434, 3673, 8376, 5682,  # 40 - 47
    6870, 4126, 8466, 5772, 6124, 3363, 6232, 3471,  # 48 - 55
    5969, 3207, 6405, 3644, 8412, 5601, 5519, 3804]


def build_face_hand_bbox(mask, joints=None):
    y, x = np.where(mask)
    # if no mask -> return no bbox
    if y.shape[0] == 0:
        return None

    # if mask is visible -> build bbox as union of joints and points on mask
    pts = np.stack([x, y], axis=1) + 0.5
    if joints is not None:
        pts = np.concatenate([pts, joints], axis=0)

    pt1, pt2 = pts.min(0), pts.max(0)

    # limit by image shape
    h, w = mask.shape
    pt1 = np.maximum(pt1, np.array([0., 0.]))
    pt2 = np.minimum(pt2, np.array([w, h]))

    bbox = np.concatenate([pt1, pt2], axis=0)

    return bbox


def build_avatar(
        root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, shape, expr,
        transl, focal, princpt, R_aug, S_aug, do_flip, gender='neutral', use_markers=False):
    """Materialize SMPL-X avatar from parameters"""
    zero_pose = torch.zeros(1, 3)
    with torch.no_grad():
        smplx_output = smpl_x.layer[gender](
            betas=torch.from_numpy(shape)[None],
            expression=torch.from_numpy(expr)[None],
            global_orient=torch.from_numpy(root_pose),
            body_pose=torch.from_numpy(body_pose).view(1, -1),
            jaw_pose=torch.from_numpy(jaw_pose).view(1, -1),
            leye_pose=zero_pose,
            reye_pose=zero_pose,
            left_hand_pose=torch.from_numpy(lhand_pose).view(1, -1),
            right_hand_pose=torch.from_numpy(rhand_pose).view(1, -1), )
    joint_cam = smplx_output.joints[0].numpy()
    pelvis = joint_cam[0]  # (3, )

    # update parameters accordingly to augmentations
    # rotation -> move rotation to root_pose, update camera transl and pp
    root_pose, _ = cv2.Rodrigues(root_pose)
    root_pose, _ = cv2.Rodrigues(np.dot(R_aug, root_pose))
    root_pose = root_pose.T

    transl = -pelvis + R_aug @ (pelvis + transl)
    princpt = R_aug[:2, :2] @ princpt

    with torch.no_grad():
        smplx_output = smpl_x.layer[gender](
            betas=torch.from_numpy(shape)[None],
            expression=torch.from_numpy(expr)[None],
            global_orient=torch.from_numpy(root_pose),
            body_pose=torch.from_numpy(body_pose).view(1, -1),
            jaw_pose=torch.from_numpy(jaw_pose).view(1, -1),
            leye_pose=zero_pose,
            reye_pose=zero_pose,
            left_hand_pose=torch.from_numpy(lhand_pose).view(1, -1),
            right_hand_pose=torch.from_numpy(rhand_pose).view(1, -1), )

    # scale -> update camera focal length and pp
    s = S_aug[[0, 1], [0, 1]]
    t = S_aug[:2, 2]
    focal = s * focal
    princpt = s * princpt + t

    joint_cam = smplx_output.joints[0].numpy()[smpl_x.joint_idx, :]
    joint_img = focal * ((joint_cam[:, :2] + transl[:2]) / (joint_cam[:, 2:] + transl[2:])) + princpt

    # create root-aligned for wrists and neck and not root_aligned versions of joints_cam
    # both are aligned by pelvis
    joint_cam_wo_ra = joint_cam - joint_cam[smpl_x.root_joint_idx, None, :]  # root-relative
    joint_cam_ra = joint_cam_wo_ra.copy()
    joint_cam_ra[smpl_x.joint_part['lhand'], :] = joint_cam_ra[smpl_x.joint_part['lhand'], :] - \
                                                  joint_cam_ra[smpl_x.lwrist_idx, None, :]  # left hand root-relative
    joint_cam_ra[smpl_x.joint_part['rhand'], :] = joint_cam_ra[smpl_x.joint_part['rhand'], :] - \
                                                  joint_cam_ra[smpl_x.rwrist_idx, None, :]  # right hand root-relative
    joint_cam_ra[smpl_x.joint_part['face'], :] = joint_cam_ra[smpl_x.joint_part['face'], :] - \
                                                 joint_cam_ra[smpl_x.neck_idx, None, :]  # face root-relative

    # for img space joints normalize x, y to the shape of heatmaps
    joint_img_orig = joint_img.copy()
    joint_img[:, 0] = joint_img[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
    joint_img[:, 1] = joint_img[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
    # for z coordinate discretize by size of heatmap
    joint_img = np.concatenate((joint_img[:, :2], joint_cam_ra[:, 2:]), 1)  # x, y, depth

    joint_img[smpl_x.joint_part['body'], 2] = (joint_img[smpl_x.joint_part['body'], 2] / (
            cfg.body_3d_size / 2) + 1) / 2. * cfg.output_hm_shape[0]  # body depth discretize

    joint_img[smpl_x.joint_part['lhand'], 2] = (joint_img[smpl_x.joint_part['lhand'], 2] / (
            cfg.hand_3d_size / 2) + 1) / 2. * cfg.output_hm_shape[0]  # left hand depth discretize

    joint_img[smpl_x.joint_part['rhand'], 2] = (joint_img[smpl_x.joint_part['rhand'], 2] / (
            cfg.hand_3d_size / 2) + 1) / 2. * cfg.output_hm_shape[0]  # right hand depth discretize

    joint_img[smpl_x.joint_part['face'], 2] = (joint_img[smpl_x.joint_part['face'], 2] / (
            cfg.face_3d_size / 2) + 1) / 2. * cfg.output_hm_shape[0]  # face depth discretize

    joint_valid = np.ones_like(joint_img[:, :1])

    joint_trunc = \
        ((joint_img[:, 0] >= 0) * (joint_img[:, 0] < cfg.output_hm_shape[2]) *
         (joint_img[:, 1] >= 0) * (joint_img[:, 1] < cfg.output_hm_shape[1]) *
         (joint_img[:, 2] >= 0) * (joint_img[:, 2] < cfg.output_hm_shape[0])).reshape(-1, 1).astype(np.float32)

    if use_markers:
        marker_cam = smplx_output.vertices[0].numpy()[markers_idxs, :]
        marker_img = focal * ((marker_cam[:, :2] + transl[:2]) / (marker_cam[:, 2:] + transl[2:])) + princpt

        # relative to pelvis
        marker_cam_wo_ra = marker_cam - joint_cam[smpl_x.root_joint_idx, None, :]  # root-relative

        # for img space joints normalize x, y to the shape of heatmaps
        marker_img[:, 0] = marker_img[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
        marker_img[:, 1] = marker_img[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
        # for z coordinate discretize by size of heatmap
        marker_img = np.concatenate((marker_img[:, :2], marker_cam_wo_ra[:, 2:]), 1)  # x, y, depth

        # body depth discretize
        marker_img[:, 2] = (marker_img[:, 2] / (cfg.body_3d_size / 2) + 1) / 2. * cfg.output_hm_shape[0]
        marker_valid = np.ones_like(marker_img[:, :1])

        marker_trunc = \
            ((marker_img[:, 0] >= 0) * (marker_img[:, 0] < cfg.output_hm_shape[2]) *
             (marker_img[:, 1] >= 0) * (marker_img[:, 1] < cfg.output_hm_shape[1]) *
             (marker_img[:, 2] >= 0) * (marker_img[:, 2] < cfg.output_hm_shape[0])).reshape(-1, 1).astype(np.float32)
    else:
        marker_cam_wo_ra, marker_img, marker_valid, marker_trunc = None, None, None, None

    return (root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, shape, expr,
            transl, focal, princpt,
            joint_cam_wo_ra, joint_cam_ra, joint_img, joint_img_orig, joint_valid, joint_trunc,
            marker_cam_wo_ra, marker_img, marker_valid, marker_trunc)


class LAIONSynth(Dataset):
    def __init__(self, transform, data_split='train'):
        super().__init__()
        images_path = '/mnt/vol_f/control-human-gen-v2/laion-faces/generations/densepose_gen_l32_shape_resampled/'
        # images_path = '/mnt/vol_f/control-human-gen-v2/laion-faces/generations/densepose_gen_h32/'
        labels_path = '/mnt/vol_f/control-human-gen-v2/laion-faces/smplerx-pred-v3-dp/'
        # labels_path = '/mnt/vol_f/control-human-gen-v2/laion-faces/smplerx-pred-h32/'
        self.images_path = images_path
        self.labels_path = labels_path
        self.data_split = data_split

        self.labels_files_list = glob.glob(os.path.join(labels_path, '*/laion_face/*/*.h5'))
        # filter only pairs where both label and image exist
        new_labels_files_list = []
        for label_path in self.labels_files_list:
            image_path = label_path.replace(labels_path, images_path).replace('.h5', '.jpg')
            if os.path.exists(image_path):
                new_labels_files_list.append(label_path)
        self.labels_files_list = new_labels_files_list

        self.use_markers = True
        self.markers_flip_pairs = (
            (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15), (16, 17), (18, 19),
            (20, 21), (22, 23), (24, 25), (26, 27), (28, 29), (30, 31), (32, 33), (34, 35), (36, 37),
            (38, 39), (40, 41), (42, 43), (44, 45), (46, 47), (48, 49), (50, 51), (52, 53), (54, 55),
            (56, 57), (58, 59), (60, 61), (62, 62), (63, 63)
        )

    def __len__(self):
        return len(self.labels_files_list)

    @staticmethod
    def process_hand_face_bbox(bbox):
        if bbox is None:
            bbox = np.array([0, 0, 1, 1], dtype=np.float32).reshape(2, 2)  # dummy value
            bbox_valid = float(False)  # dummy value
        else:
            # reshape to top-left (x,y) and bottom-right (x,y)
            bbox = bbox.reshape(2, 2)

            bbox[:, 0] = bbox[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
            bbox[:, 1] = bbox[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
            bbox_valid = float(True)

        return bbox, bbox_valid

    def __getitem__(self, idx):
        label_path = self.labels_files_list[idx]
        image_path = label_path.replace(self.labels_path, self.images_path).replace('.h5', '.jpg')

        img_orig = cv2.imread(image_path)[..., ::-1].copy()
        img_shape = img_orig.shape[:2]
        avatar_meta = dd.io.load(label_path)

        # get bbox transform and augmentations
        bbox = avatar_meta['bbox']
        img, img2bb_trans, bb2img_trans, rot, do_flip, R_aug, S_aug = augmentation(
            img_orig, bbox, self.data_split, preserve_aspect_ratio=True)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.

        # smplx parameters
        root_pose = avatar_meta['global_orient']  # rotation to world coordinate
        body_pose = avatar_meta['body_pose']
        lhand_pose = avatar_meta['left_hand_pose']
        rhand_pose = avatar_meta['right_hand_pose']
        jaw_pose = avatar_meta['jaw_pose']
        shape = avatar_meta['betas_shape_resampled']
        expr = avatar_meta['expression']
        transl = avatar_meta['transl']  # translation to world coordinate

        focal, princpt = avatar_meta['focal'], avatar_meta['princpt']

        (root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, smplx_shape, smplx_expr,
         transl, focal, princpt, joint_cam_wo_ra, joint_cam_ra, joint_img, joint_img_orig, joint_valid, joint_trunc,
         marker_cam, marker_img, marker_valid, marker_trunc) = build_avatar(
            root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, shape, expr,
            transl, focal, princpt, R_aug, S_aug, do_flip, use_markers=self.use_markers)

        # build bounding boxes for hands and face
        densepose = cv2.imdecode(avatar_meta['smplx_densepose_shape_resampled'], cv2.IMREAD_GRAYSCALE)
        densepose = cv2.warpAffine(densepose, img2bb_trans, cfg.input_img_shape[::-1], flags=cv2.INTER_NEAREST)

        lhand_bbox = build_face_hand_bbox(densepose == 3, joint_img_orig[smpl_x.joint_part['lhand'], :])
        rhand_bbox = build_face_hand_bbox(densepose == 2, joint_img_orig[smpl_x.joint_part['rhand'], :])
        face_bbox = build_face_hand_bbox(densepose == 14, joint_img_orig[smpl_x.joint_part['face'], :])

        lhand_bbox, lhand_bbox_valid = self.process_hand_face_bbox(lhand_bbox)
        rhand_bbox, rhand_bbox_valid = self.process_hand_face_bbox(rhand_bbox)
        face_bbox, face_bbox_valid = self.process_hand_face_bbox(face_bbox)

        if do_flip:
            lhand_bbox, rhand_bbox = rhand_bbox, lhand_bbox
            lhand_bbox_valid, rhand_bbox_valid = rhand_bbox_valid, lhand_bbox_valid
        lhand_bbox_center = (lhand_bbox[0] + lhand_bbox[1]) / 2.
        rhand_bbox_center = (rhand_bbox[0] + rhand_bbox[1]) / 2.
        face_bbox_center = (face_bbox[0] + face_bbox[1]) / 2.
        lhand_bbox_size = lhand_bbox[1] - lhand_bbox[0]
        rhand_bbox_size = rhand_bbox[1] - rhand_bbox[0]
        face_bbox_size = face_bbox[1] - face_bbox[0]

        smplx_pose = np.concatenate((root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose))

        inputs = {'img': img, 'focal': focal[None], 'princpt': princpt[None]}
        targets = {
            'joint_img': joint_img, 'joint_cam': joint_cam_wo_ra,
            'smplx_joint_img': joint_img, 'smplx_joint_cam': joint_cam_ra,
            'smplx_pose': smplx_pose.reshape(-1),
            'smplx_shape': smplx_shape.reshape(-1),
            'smplx_expr': smplx_expr.reshape(-1),
            'lhand_bbox_center': lhand_bbox_center, 'lhand_bbox_size': lhand_bbox_size,
            'rhand_bbox_center': rhand_bbox_center, 'rhand_bbox_size': rhand_bbox_size,
            'face_bbox_center': face_bbox_center, 'face_bbox_size': face_bbox_size,
            'trans': transl
        }
        meta_info = {
            'joint_valid': joint_valid, 'joint_trunc': joint_trunc,
            'smplx_joint_valid': joint_valid, 'smplx_joint_trunc': joint_trunc,
            'smplx_pose_valid': float(True), 'smplx_shape_valid': float(True),
            'smplx_expr_valid': float(True), 'is_3D': float(True),
            'lhand_bbox_valid': lhand_bbox_valid,
            'rhand_bbox_valid': rhand_bbox_valid,
            'face_bbox_valid': face_bbox_valid}

        if self.use_markers:
            targets = {**targets, **{'marker_img': marker_img, 'marker_cam': marker_cam,
                                     'smplx_marker_img': marker_img}}
            meta_info = {**meta_info, **{'marker_valid': marker_valid, 'marker_trunc': marker_trunc,
                                         'smplx_marker_valid': marker_valid, 'smplx_marker_trunc': marker_trunc, }}

        return inputs, targets, meta_info
