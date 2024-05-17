import os

import cv2
import numpy as np
import smplx
import torch


class SSP3DDataset(torch.utils.data.Dataset):
    AVATAR_TYPE = 'SMPL'
    HAS_GT_BBOX = True

    def __init__(self, data_path, smpl_model_path, return_vertices: bool = False, return_projection: bool = False,
                 zero_pose: bool = False):
        super().__init__()
        self.data_path = data_path
        self.smpl_model_path = smpl_model_path
        self.return_vertices = return_vertices
        self.return_projection = return_projection
        self.zero_pose = zero_pose

        metadata_path = os.path.join(data_path, 'labels.npz')
        self.metadata = np.load(metadata_path)

        device = 'cpu'
        smpl_neutral = smplx.create(
            os.path.join(smpl_model_path, 'SMPL_NEUTRAL.pkl'),
            model_type='smpl', gender='neutral', num_betas=10,
        ).to(device)

        smpl_male = smplx.create(
            os.path.join(smpl_model_path, 'SMPL_MALE.pkl'),
            model_type='smpl', gender='male', num_betas=10,
        ).to(device)

        smpl_female = smplx.create(
            os.path.join(smpl_model_path, 'SMPL_FEMALE.pkl'),
            model_type='smpl', gender='female', num_betas=10,
        ).to(device)

        self.smpl_models = dict(
            n=smpl_neutral,
            m=smpl_male,
            f=smpl_female
        )
        self.f = np.array([5000, 5000], dtype=np.float32)

    def __len__(self):
        return len(self.metadata['fnames'])

    def __getitem__(self, idx):
        image_path = self.metadata['fnames'][idx]
        image_path = os.path.join(self.data_path, 'images', image_path)

        img = cv2.imread(image_path)[..., ::-1]
        gender = self.metadata['genders'][idx]
        smpl_model = self.smpl_models[gender]
        with torch.no_grad():
            if self.zero_pose:
                smpl_output = smpl_model(
                    betas=torch.from_numpy(self.metadata['shapes'][idx])[None],
                    global_orient=torch.zeros(1, 1, 3),
                    body_pose=torch.zeros(1, 23, 3),
                    transl=torch.from_numpy(self.metadata['cam_trans'][idx])[None].float()
                )
            else:
                smpl_output = smpl_model(
                    betas=torch.from_numpy(self.metadata['shapes'][idx])[None],
                    global_orient=torch.from_numpy(self.metadata['poses'][idx, :3])[None, None].float(),
                    body_pose=torch.from_numpy(self.metadata['poses'][idx, 3:]).view(1, 23, 3),
                    transl=torch.from_numpy(self.metadata['cam_trans'][idx])[None].float()
                )
        vertices = smpl_output.vertices[0].cpu().numpy()

        # align root joint for 3d vertices
        pelvis = smpl_model.J_regressor[0:1, :].numpy() @ vertices
        vertices_aligned = vertices - pelvis

        H, W, _ = img.shape
        pp = np.array([W, H], dtype=np.float32) / 2

        v2 = vertices[:, :2] / vertices[:, 2:]
        v2 = v2 * self.f + pp

        # find_bbox
        x1, y1 = np.min(v2, axis=0)
        x2, y2 = np.max(v2, axis=0)
        bbox = np.array([x1, y1, x2 - x1, y2 - y1])

        return img, bbox, vertices_aligned, v2
