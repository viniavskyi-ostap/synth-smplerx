import os

import deepdish as dd
import numpy as np
import smplx
import torch
from tqdm import tqdm

from eval.egobody import EgoBodyDataset
from eval.ehf import EHFDataset
from eval.ssp3d import SSP3DDataset
from eval.ubody import UBodyDataset


def pve(vertices_gt, vertices_pred):
    return np.linalg.norm(vertices_gt - vertices_pred, axis=1).mean() * 1000


def rigid_transform_3D(A, B):
    n, dim = A.shape
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B) / n
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        s[-1] = -s[-1]
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))

    varP = np.var(A, axis=0).sum()
    c = 1 / varP * np.sum(s)

    t = -np.dot(c * R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return c, R, t


def rigid_align(A, B):
    c, R, t = rigid_transform_3D(A, B)
    A2 = np.transpose(np.dot(c * R, np.transpose(A))) + t
    return A2


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process a folder.")
    parser.add_argument("--dataset", choices=["ehf", "egobody", "ssp3d", "ssp3d_shape", "ubody"])
    parser.add_argument("--experiment_tag", type=str)
    parser.add_argument("--device", default="cpu", type=str)
    args = parser.parse_args()

    args = parser.parse_args()

    zero_pred_pose = False
    if args.dataset == "ehf":
        dataset = EHFDataset(
            data_path='/mnt/vol_f/smplify_cse/ehf',
            smplx_model_path='/mnt/vol_d/projects/SMPLer-X/common/utils/human_model_files/smplx/',
            return_vertices=True
        )
    elif args.dataset == "egobody":
        dataset = EgoBodyDataset(
            data_path='/mnt/vol_f/datasets/EgoBody',
            smpl_model_path='/mnt/vol_d/projects/SMPLer-X/common/utils/human_model_files/smpl/',
            return_vertices=True
        )
    elif args.dataset == "ssp3d":
        dataset = SSP3DDataset(
            data_path='/mnt/vol_f/smplify_cse/ssp3d/',
            smpl_model_path='/mnt/vol_d/projects/SMPLer-X/common/utils/human_model_files/smpl/',
            return_vertices=True
        )
    elif args.dataset == "ssp3d_shape":
        dataset = SSP3DDataset(
            data_path='/mnt/vol_f/smplify_cse/ssp3d/',
            smpl_model_path='/mnt/vol_d/projects/SMPLer-X/common/utils/human_model_files/smpl/',
            return_vertices=True,
            zero_pose=True
        )
        zero_pred_pose = True
        args.dataset = "ssp3d"
    elif args.dataset == "ubody":
        dataset = UBodyDataset(
            data_path="/mnt/vol_f/datasets",
            smplx_model_path="/mnt/vol_c/projects/smplx-estimation-bh/weights/smplx",
            return_vertices=True,
        )
    else:
        raise ValueError

    device = args.device
    smplx_model_path = '/mnt/vol_d/projects/SMPLer-X/common/utils/human_model_files/smplx/'
    smpl_model_path = '/mnt/vol_d/projects/SMPLer-X/common/utils/human_model_files/smpl/'

    smplx_neutral = smplx.create(
        os.path.join(smplx_model_path, 'SMPLX_NEUTRAL.npz'),
        model_type='smplx', gender='neutral', num_betas=10,
        use_face_contour=False,
        flat_hand_mean=False,
        num_expression_coeffs=10,
        ext='npz', use_pca=False
    ).to(device)
    smpl_neutral = smplx.create(
        os.path.join(smpl_model_path, 'SMPL_NEUTRAL.pkl'),
        model_type='smpl', gender='neutral', num_betas=10,
    ).to(device)

    deformation_transfer = dd.io.load('/mnt/vol_d/projects/SMPLer-X/smplx2smpl_sparse.h5')
    deformation_transfer = {k: torch.from_numpy(v).to(device) for k, v in deformation_transfer.items()}

    predictions_path = f"prediction/{args.dataset}/{args.experiment_tag}"

    pve_metric, papve_metric = [], []

    for i in tqdm(range(len(dataset))):
        instance = dataset[i]
        if dataset.HAS_GT_BBOX:
            image, _, vertices_gt, *_ = instance
        else:
            image, vertices_gt, *_ = instance

        # read prediction
        avatar_info = dd.io.load(os.path.join(predictions_path, f'{i}.h5'))
        with torch.no_grad():
            if zero_pred_pose:
                smplx_output = smplx_neutral(
                    betas=torch.from_numpy(avatar_info['betas']).to(device)[None],
                    expression=torch.from_numpy(avatar_info['expression']).to(device)[None],
                    global_orient=torch.zeros(1, 1, 3, device=device),
                    body_pose=torch.zeros(1, 21, 3, device=device),
                    jaw_pose=torch.zeros(1, 1, 3, device=device),
                    leye_pose=torch.zeros(1, 1, 3, device=device),
                    reye_pose=torch.zeros(1, 1, 3, device=device),
                    left_hand_pose=torch.zeros(1, 15, 3, device=device),
                    right_hand_pose=torch.zeros(1, 15, 3, device=device),
                )
            else:
                smplx_output = smplx_neutral(
                    betas=torch.from_numpy(avatar_info['betas']).to(device)[None],
                    expression=torch.from_numpy(avatar_info['expression']).to(device)[None],
                    global_orient=torch.from_numpy(avatar_info['global_orient']).to(device)[None],
                    body_pose=torch.from_numpy(avatar_info['body_pose']).to(device)[None],
                    jaw_pose=torch.from_numpy(avatar_info['jaw_pose']).to(device)[None],
                    leye_pose=torch.from_numpy(avatar_info['leye_pose']).to(device)[None],
                    reye_pose=torch.from_numpy(avatar_info['reye_pose']).to(device)[None],
                    left_hand_pose=torch.from_numpy(avatar_info['left_hand_pose']).to(device)[None],
                    right_hand_pose=torch.from_numpy(avatar_info['right_hand_pose']).to(device)[None],
                )
        vertices_pred = smplx_output.vertices[0]

        if dataset.AVATAR_TYPE == "SMPL":
            # apply transfer from predicted SMPL-X to SMPL
            vertices_pred = (
                    vertices_pred[deformation_transfer['idxs']] * deformation_transfer['vals'][..., None]).sum(1)
            pelvis_pred = smpl_neutral.J_regressor[0:1, :] @ vertices_pred
        else:
            pelvis_pred = smplx_neutral.J_regressor[0:1, :] @ vertices_pred
        vertices_pred = (vertices_pred - pelvis_pred).numpy()

        pve_value = pve(vertices_pred, vertices_gt)
        vertices_pred_pa = rigid_align(vertices_pred, vertices_gt)
        papve_value = pve(vertices_pred_pa, vertices_gt)

        pve_metric.append(pve_value)
        papve_metric.append(papve_value)

        print(f'Image {i}: PVE: {pve_value}, PAPVE: {papve_value}')

    print('===========')
    print(f'Dataset: {args.dataset}, Method: {args.experiment_tag}')
    print(f'PVE: {np.mean(pve_metric)}, PAPVE: {np.mean(papve_metric)}')
