import os

import cv2
import numpy as np
import smplx
import torch


class EgoBodyDataset(torch.utils.data.Dataset):
    TEST_SAMPLE_INTERVAL = 100
    AVATAR_TYPE = 'SMPL'
    HAS_GT_BBOX = True

    def __init__(self, data_path, smpl_model_path, return_vertices: bool = False, return_projection: bool = False):
        super().__init__()
        self.data_path = data_path
        self.smpl_model_path = smpl_model_path
        self.return_vertices = return_vertices
        self.return_projection = return_projection

        metadata_path = os.path.join(data_path, 'egocapture_test_smpl.npz')
        self.metadata = np.load(metadata_path)
        self.metadata = {k: self.metadata[k] for k in self.metadata.files}

        #  downsample
        for k in self.metadata.keys():
            self.metadata[k] = self.metadata[k][::self.TEST_SAMPLE_INTERVAL]

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

    def __len__(self):
        return len(self.metadata['imgname'])

    def __getitem__(self, idx):
        image_path = self.metadata['imgname'][idx]
        # _, recording_id, _, _, img_fname = image_path.split('/')
        # _, _, frame_id = img_fname[:-4].split('_')
        image_path = os.path.join(self.data_path, image_path)
        # labels_path = glob.glob(os.path.join(self.data_path, 'smplx_interactee_test', recording_id, 'body_idx_*', 'results', f'frame_{frame_id}', '000.pkl'))[0]

        img = cv2.imread(image_path)[..., ::-1]

        with torch.no_grad():
            smpl_output = self.smpl_models[self.metadata['gender'][idx]](
                betas=torch.from_numpy(self.metadata['betas'][idx])[None],
                global_orient=torch.from_numpy(self.metadata['global_orient_pv'][idx])[None, None].float(),
                body_pose=torch.from_numpy(self.metadata['body_pose'][idx]).view(1, 23, 3),
                transl=torch.from_numpy(self.metadata['transl_pv'][idx])[None].float()
            )
        vertices = smpl_output.vertices[0].cpu().numpy()

        # align root joint for 3d vertices
        pelvis = self.smpl_models[self.metadata['gender'][idx]].J_regressor[0:1, :].numpy() @ vertices
        vertices_aligned = vertices - pelvis


        pp = np.array([self.metadata['cx'][idx], self.metadata['cy'][idx]], dtype=np.float32)
        f = np.array([self.metadata['fx'][idx], self.metadata['fy'][idx]], dtype=np.float32)

        v2 = vertices[:, :2] / vertices[:, 2:]
        v2 = v2 * f + pp

        # find_bbox
        x1, y1 = np.min(v2, axis=0)
        x2, y2 = np.max(v2, axis=0)
        bbox = np.array([x1, y1, x2 - x1, y2 - y1])

        return img, bbox, vertices_aligned, v2
