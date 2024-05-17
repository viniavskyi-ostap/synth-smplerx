import glob
import os

import cv2
import numpy as np
import smplx
import torch
import trimesh


class EHFDataset(torch.utils.data.Dataset):
    AVATAR_TYPE = 'SMPLX'
    HAS_GT_BBOX = True

    def __init__(self, data_path, smplx_model_path, return_vertices: bool = False, return_projection: bool = False):
        super().__init__()
        self.data_path = data_path
        self.smplx_model_path = smplx_model_path
        self.return_vertices = return_vertices
        self.return_projection = return_projection

        self.images_list = sorted(glob.glob(os.path.join(self.data_path, '*_img.png')))

        rot_vec = np.array(
            [[-2.98747896],
             [0.01172457],
             [-0.05704687]], dtype=np.float32)

        self.R, _ = cv2.Rodrigues(rot_vec)
        self.T = np.array([-0.03609917, 0.43416458, 2.37101226], dtype=np.float32)
        self.f = np.array([1498.22426237, 1498.22426237], dtype=np.float32)
        self.pp = np.array([790.263706, 578.90334], dtype=np.float32)

        device = 'cpu'
        self.smplx_neutral = smplx.create(
            os.path.join(smplx_model_path, 'SMPLX_NEUTRAL.npz'),
            model_type='smplx', gender='neutral', num_betas=10,
            use_face_contour=False,
            flat_hand_mean=False,
            num_expression_coeffs=10,
            ext='npz', use_pca=False
        ).to(device)

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image_path = self.images_list[idx]
        img = cv2.imread(image_path)[..., ::-1]

        image_id = image_path.rsplit('/', 1)[-1].split('_')[0]

        vertices = np.array(
            trimesh.load(os.path.join(self.data_path, f'{image_id}_align.ply')).vertices, dtype=np.float32)
        vertices = vertices @ self.R.T + self.T

        # align root joint for 3d vertices
        pelvis = self.smplx_neutral.J_regressor[0:1, :].numpy() @ vertices
        vertices_aligned = vertices - pelvis

        # if self.return_projection:
        v2 = vertices[:, :2] / vertices[:, 2:]
        v2 = v2 * self.f + self.pp

        # find_bbox
        x1, y1 = np.min(v2, axis=0)
        x2, y2 = np.max(v2, axis=0)
        bbox = np.array([x1, y1, x2 - x1, y2 - y1])

        return img, bbox, vertices_aligned, v2
