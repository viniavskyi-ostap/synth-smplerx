import os
import cv2
import json
import smplx
import torch
import deepdish as dd
import numpy as np

import os.path as osp

from pycocotools.coco import COCO


class UBodyDatasetPart(torch.utils.data.Dataset):
    TEST_SAMPLE_INTERVAL = 1000

    def __init__(
            self, data_path, smplx_model_path, scene,
            return_vertices: bool = False,
            return_projection: bool = False
    ) -> None:
        super().__init__()

        self.data_path = data_path

        self.img_path = osp.join(self.data_path, 'UBody', 'images', scene)
        self.annot_path = osp.join(self.data_path, 'UBody', 'annotations', scene, 'keypoint_annotation.json')
        self.smplx_annot_path = osp.join(self.data_path, 'UBody', 'annotations', scene, 'smplx_annotation.json')
        self.test_video_list_path = osp.join(self.data_path, 'UBody', 'splits', 'intra_scene_test_list.npy')

        self.smplx_model_path = smplx_model_path
        self.smplx_model = smplx.SMPLX(smplx_model_path, num_betas=10, use_pca=False)

        self.return_vertices = return_vertices
        self.return_projection = return_projection

        cache_path = osp.join(self.data_path, 'UBody', 'cache', scene, 'annos.h5')
        if osp.exists(cache_path):
            self.annos = dd.io.load(cache_path)
        else:
            self.annos = self.load_annos()
            os.makedirs(osp.dirname(cache_path), exist_ok=True)
            dd.io.save(cache_path, self.annos)

    def load_annos(self):
        db = COCO(self.annot_path)
        with open(self.smplx_annot_path) as f:
            smplx_params_annos = json.load(f)

        annos = []

        i = 0
        for aid in db.anns.keys():
            i = i + 1
            if i % self.TEST_SAMPLE_INTERVAL != 0:
                continue

            ann = db.anns[aid]
            if ann['valid_label'] == 0 or str(aid) not in smplx_params_annos:
                continue

            img = db.loadImgs(ann['image_id'])[0]
            bbox = ann["bbox"]

            if bbox is None:
                continue

            smplx_param = smplx_params_annos[str(aid)]

            annos.append((
                img, bbox, smplx_param
            ))

        return annos

    def __len__(self) -> int:
        return len(self.annos)

    def __getitem__(self, idx: int):
        img_meta, bbox, smplx_param = self.annos[idx]

        img = cv2.imread(
            os.path.join(self.img_path, img_meta["file_name"]),
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )[:, :, ::-1]

        with torch.no_grad():
            smplx_output = self.smplx_model(
                betas=torch.tensor(smplx_param["smplx_param"]["shape"], dtype=torch.float32).unsqueeze(0),
                expression=torch.tensor(smplx_param["smplx_param"]["expr"], dtype=torch.float32).unsqueeze(0),
                body_pose=torch.tensor(smplx_param["smplx_param"]["body_pose"], dtype=torch.float32).unsqueeze(0),
                left_hand_pose=torch.tensor(smplx_param["smplx_param"]["lhand_pose"], dtype=torch.float32).unsqueeze(0),
                right_hand_pose=torch.tensor(smplx_param["smplx_param"]["rhand_pose"], dtype=torch.float32).unsqueeze(0),
                jaw_pose=torch.tensor(smplx_param["smplx_param"]["jaw_pose"], dtype=torch.float32).unsqueeze(0),
                global_orient=torch.tensor(smplx_param["smplx_param"]["root_pose"], dtype=torch.float32).unsqueeze(0),
                transl=torch.tensor(smplx_param["smplx_param"]["trans"], dtype=torch.float32).unsqueeze(0),
            )

        vertices = smplx_output.vertices[0].cpu().numpy()

        # align root joint for 3d vertices
        pelvis = self.smplx_model.J_regressor[0:1, :].numpy() @ vertices
        vertices_aligned = vertices - pelvis

        assert not ("R" in smplx_param["cam_param"])
        pp = np.array(smplx_param["cam_param"]["princpt"], dtype=np.float32)
        f = np.array(smplx_param["cam_param"]["focal"], dtype=np.float32)

        v2 = vertices[:, :2] / vertices[:, 2:]
        v2 = v2 * f + pp

        return img, np.array(bbox), vertices_aligned, v2


class UBodyDataset(torch.utils.data.Dataset):
    AVATAR_TYPE = 'SMPLX'
    HAS_GT_BBOX = True

    def __init__(
            self, data_path, smplx_model_path,
            return_vertices: bool = False,
            return_projection: bool = False
    ) -> None:
        scene_names = os.listdir(os.path.join(data_path, 'UBody', 'images'))
        dbs = []
        for scene_name in scene_names:
            dbs.append(
                UBodyDatasetPart(
                    data_path=data_path,
                    smplx_model_path=smplx_model_path,
                    return_vertices=return_vertices,
                    return_projection=return_projection,
                    scene=scene_name
                )
            )

        self.dbs = dbs
        self.db_num = len(self.dbs)
        self.max_db_data_num = max([len(db) for db in dbs])
        self.db_len_cumsum = np.cumsum([len(db) for db in dbs])

    def __len__(self):
        return sum([len(db) for db in self.dbs])

    def __getitem__(self, index):
        for i in range(self.db_num):
            if index < self.db_len_cumsum[i]:
                db_idx = i
                break

        if db_idx == 0:
            data_idx = index
        else:
            data_idx = index - self.db_len_cumsum[db_idx - 1]

        return self.dbs[db_idx][data_idx]


if __name__ == "__main__":
    dataset = UBodyDataset(
        data_path="/mnt/vol_f/datasets",
        smplx_model_path="/mnt/vol_c/projects/smplx-estimation-bh/weights/smplx",
        return_vertices=True,
        return_projection=True,
    )