import os
import os.path as osp
from typing import List, Optional

import cv2
import numpy as np
from torch.utils.data import Dataset

from vision3d.array_ops import (
    apply_transform,
    compose_transforms,
    get_2d3d_correspondences_mutual,
    get_transform_from_rotation_translation,
    inverse_transform,
    random_sample_small_transform,
)
from vision3d.utils.collate import GraphPyramid2D3DRegistrationCollateFn
from vision3d.utils.dataloader import build_dataloader, calibrate_neighbors_pack_mode


def load_calib(calib_path):
    """Load KITTI-style calibration. Returns (4,4) T_cam_from_lidar.

    Expected format:
        Tr: r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz
    """
    with open(calib_path, "r") as f:
        line = f.readline().strip()
    values = [float(x) for x in line.split()[1:]]  # skip "Tr:"
    T = np.eye(4, dtype=np.float64)
    T[:3, :] = np.array(values).reshape(3, 4)
    return T


def project_to_depth(points, transform, intrinsics, image_h, image_w):
    """Project 3D points (N, 3) in the point cloud frame into a sparse depth map.

    Args:
        points (ndarray): (N, 3) float32, point cloud xyz in LiDAR frame.
        transform (ndarray): (4, 4) float32, T_cam_from_lidar @ inv(gt).
        intrinsics (ndarray): (3, 3) float32, camera intrinsics.
        image_h, image_w (int): target image resolution.

    Returns:
        depth (ndarray): (H, W) float32, sparse depth map in metres.
    """
    if points.shape[0] == 0:
        return np.zeros((image_h, image_w), dtype=np.float32)

    pts_hom = np.concatenate(
        [points, np.ones((len(points), 1), dtype=np.float32)], axis=1
    )  # (N, 4)
    pts_cam = (transform @ pts_hom.T).T[:, :3]  # (N, 3)

    valid = pts_cam[:, 2] > 0.1
    pts_cam = pts_cam[valid]
    if pts_cam.shape[0] == 0:
        return np.zeros((image_h, image_w), dtype=np.float32)

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    u = np.round(pts_cam[:, 0] / pts_cam[:, 2] * fx + cx).astype(int)
    v = np.round(pts_cam[:, 1] / pts_cam[:, 2] * fy + cy).astype(int)
    z = pts_cam[:, 2]

    in_bounds = (u >= 0) & (u < image_w) & (v >= 0) & (v < image_h)
    u, v, z = u[in_bounds], v[in_bounds], z[in_bounds]

    depth = np.zeros((image_h, image_w), dtype=np.float32)
    # sort far-to-near so nearer points overwrite farther ones
    # store in millimetres to match vision3d's back_project(scaling_factor=1000)
    order = np.argsort(-z)
    depth[v[order], u[order]] = z[order] * 1000.0
    return depth


class R3LiveDataset(Dataset):
    """Dataset for r3live-style rosbag exports organised as sequences.

    Directory layout::

        <dataset_dir>/          # e.g. data/r3live/sequences/
            00/
                calib.txt       # KITTI Tr: line (LiDAR→camera extrinsic)
                img/
                    000000.npy  # (H, W, 3) uint8 BGR
                    ...
                K/
                    000000.npy  # (3, 3) float32 intrinsics
                    ...
                pc_with_normal/
                    000000.npy  # (7, N) float32 [x,y,z,intensity,nx,ny,nz]
                    ...
                gt/
                    000000.npy  # (4, 4) float64  gt = T_j^{-1} @ T_i
            01/
                ...

    Transform convention
    --------------------
    ``gt = T_j^{-1} @ T_i`` maps from the robot body frame at image time *i*
    to the body frame at point-cloud time *j*.
    The model needs ``transform`` that maps LiDAR points (at time *j*) into
    the camera frame (at time *i*)::

        transform = T_cam_from_lidar @ inv(gt)

    Sequence split
    --------------
    Pass explicit sequence names via ``sequences``.
    When ``sequences`` is None, all subdirectories found under ``dataset_dir``
    are sorted and split 70% / 15% / 15% chronologically.
    """

    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15

    def __init__(
        self,
        dataset_dir: str,
        subset: str,
        sequences: Optional[List[str]] = None,
        max_points: Optional[int] = None,
        return_corr_indices: bool = False,
        matching_radius_2d: float = 8.0,
        matching_radius_3d: float = 0.075,
        overlap_threshold: Optional[float] = None,
        use_augmentation: bool = False,
        augmentation_noise: float = 0.005,
        image_h: int = 480,
        image_w: int = 640,
    ):
        super().__init__()
        assert subset in ["train", "val", "test"], f"Bad subset: {subset}"
        assert image_h % 24 == 0, f"image_h ({image_h}) must be divisible by 24"
        assert image_w % 32 == 0, f"image_w ({image_w}) must be divisible by 32"

        self.dataset_dir = dataset_dir
        self.subset = subset
        self.max_points = max_points
        self.return_corr_indices = return_corr_indices
        self.matching_radius_2d = matching_radius_2d
        self.matching_radius_3d = matching_radius_3d
        self.overlap_threshold = overlap_threshold
        self.use_augmentation = use_augmentation
        self.aug_noise = augmentation_noise
        self.image_h = image_h
        self.image_w = image_w

        # ── resolve sequence list ────────────────────────────────────────────
        if sequences is not None:
            seq_names = list(sequences)
        else:
            # auto-detect: subdirectories that contain an img/ folder
            all_seqs = sorted(
                d for d in os.listdir(dataset_dir)
                if osp.isdir(osp.join(dataset_dir, d, "img"))
            )
            n = len(all_seqs)
            train_end = int(n * self.TRAIN_RATIO)
            val_end = int(n * (self.TRAIN_RATIO + self.VAL_RATIO))
            if subset == "train":
                seq_names = all_seqs[:train_end]
            elif subset == "val":
                seq_names = all_seqs[train_end:val_end]
            else:
                seq_names = all_seqs[val_end:]

        # ── build sample list and per-sequence calibrations ──────────────────
        # self.samples : list of (seq_name, frame_idx)
        # self.calibs  : dict  seq_name -> (4,4) float32 T_cam_from_lidar
        self.samples = []
        self.calibs = {}
        for seq in seq_names:
            seq_dir = osp.join(dataset_dir, seq)
            self.calibs[seq] = load_calib(
                osp.join(seq_dir, "calib.txt")
            ).astype(np.float32)
            frame_ids = sorted(
                int(f[:-4])
                for f in os.listdir(osp.join(seq_dir, "img"))
                if f.endswith(".npy")
            )
            for fid in frame_ids:
                self.samples.append((seq, fid))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        seq, idx = self.samples[index]
        seq_dir = osp.join(self.dataset_dir, seq)
        T_cam_from_lidar = self.calibs[seq]

        data_dict = {}
        data_dict["scene_name"] = seq
        data_dict["image_id"] = f"{idx:06d}"
        data_dict["cloud_id"] = f"{idx:06d}"
        data_dict["image_file"] = f"{seq}/img/{idx:06d}.npy"
        data_dict["depth_file"] = ""
        data_dict["cloud_file"] = f"{seq}/pc_with_normal/{idx:06d}.npy"
        data_dict["overlap"] = 1.0  # overlap is unknown; set to 1.0

        # ── load image ──────────────────────────────────────────────────────
        img_bgr = np.load(osp.join(seq_dir, "img", f"{idx:06d}.npy"))
        orig_h, orig_w = img_bgr.shape[:2]
        image_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if orig_h != self.image_h or orig_w != self.image_w:
            image_gray = cv2.resize(
                image_gray, (self.image_w, self.image_h), interpolation=cv2.INTER_LINEAR
            )

        # ── load & rescale intrinsics ────────────────────────────────────────
        intrinsics = np.load(
            osp.join(seq_dir, "K", f"{idx:06d}.npy")
        ).astype(np.float32)
        if orig_h != self.image_h or orig_w != self.image_w:
            intrinsics = intrinsics.copy()
            intrinsics[0] *= self.image_w / orig_w  # scale fx, cx
            intrinsics[1] *= self.image_h / orig_h  # scale fy, cy

        # ── load point cloud (7, N) → (N, 3) xyz ────────────────────────────
        pc_data = np.load(
            osp.join(seq_dir, "pc_with_normal", f"{idx:06d}.npy")
        )  # (7, N)
        points = pc_data[:3, :].T.astype(np.float32)  # (N, 3)

        # ── load gt transform (4, 4): gt = T_j^{-1} @ T_i ───────────────────
        gt = np.load(
            osp.join(seq_dir, "gt", f"{idx:06d}.npy")
        ).astype(np.float32)

        # transform from pcd (LiDAR at j) to camera (at i):
        #   p_cam = T_cam_from_lidar @ T_i^{-1} @ T_j @ p_lidar_j
        #         = T_cam_from_lidar @ inv(gt) @ p_lidar_j
        transform = (T_cam_from_lidar @ np.linalg.inv(gt)).astype(np.float32)

        # ── frustum crop: keep only points visible in the image ─────────────
        pts_hom = np.concatenate(
            [points, np.ones((len(points), 1), dtype=np.float32)], axis=1
        )
        pts_cam_pre = (transform @ pts_hom.T).T[:, :3]
        fx0, fy0 = intrinsics[0, 0], intrinsics[1, 1]
        cx0, cy0 = intrinsics[0, 2], intrinsics[1, 2]
        z_ok = pts_cam_pre[:, 2] > 0.1
        u0 = pts_cam_pre[:, 0] / np.maximum(pts_cam_pre[:, 2], 1e-6) * fx0 + cx0
        v0 = pts_cam_pre[:, 1] / np.maximum(pts_cam_pre[:, 2], 1e-6) * fy0 + cy0
        frustum_mask = z_ok & (u0 >= 0) & (u0 < self.image_w) & (v0 >= 0) & (v0 < self.image_h)
        if frustum_mask.sum() > 10:
            points = points[frustum_mask]

        # ── sub-sample point cloud ───────────────────────────────────────────
        if self.max_points is not None and points.shape[0] > self.max_points:
            sel = np.random.permutation(points.shape[0])[: self.max_points]
            points = points[sel]

        # ── augmentation ─────────────────────────────────────────────────────
        if self.use_augmentation:
            aug_transform = random_sample_small_transform()
            center = points.mean(axis=0)
            sub_center = get_transform_from_rotation_translation(None, -center)
            add_center = get_transform_from_rotation_translation(None, center)
            aug_transform = compose_transforms(sub_center, aug_transform, add_center)
            points = apply_transform(points, aug_transform)
            inv_aug = inverse_transform(aug_transform)
            transform = compose_transforms(inv_aug, transform)
            points += (np.random.rand(points.shape[0], 3) - 0.5) * self.aug_noise
            # random_sample_small_transform() returns float64; cast back to float32
            points = points.astype(np.float32)
            transform = transform.astype(np.float32)

        # ── synthetic depth from point cloud projection ──────────────────────
        depth = project_to_depth(points, transform, intrinsics, self.image_h, self.image_w)

        # ── correspondences ──────────────────────────────────────────────────
        if self.return_corr_indices:
            img_corr_pixels, pcd_corr_indices = get_2d3d_correspondences_mutual(
                depth, points, intrinsics, transform,
                self.matching_radius_2d, self.matching_radius_3d,
                depth_limit=50.0,  # outdoor: 50 m (depth stored in mm; lib divides by 1000)
            )
            img_corr_indices = img_corr_pixels[:, 0] * self.image_w + img_corr_pixels[:, 1]
            data_dict["img_corr_pixels"] = img_corr_pixels
            data_dict["img_corr_indices"] = img_corr_indices
            data_dict["pcd_corr_indices"] = pcd_corr_indices

        image_gray -= image_gray.mean()

        data_dict["image_h"] = self.image_h
        data_dict["image_w"] = self.image_w
        data_dict["intrinsics"] = intrinsics
        data_dict["transform"] = transform
        data_dict["image"] = image_gray
        data_dict["depth"] = depth
        data_dict["points"] = points
        data_dict["feats"] = np.ones((points.shape[0], 1), dtype=np.float32)

        return data_dict


def _make_collate_fn(cfg, neighbor_limits):
    return GraphPyramid2D3DRegistrationCollateFn(
        cfg.model.point_backbone.num_stages,
        cfg.model.point_backbone.base_voxel_size,
        cfg.model.point_backbone.base_voxel_size * cfg.model.point_backbone.kpconv_radius,
        neighbor_limits,
    )


def train_valid_data_loader(cfg):
    train_dataset = R3LiveDataset(
        cfg.data.dataset_dir,
        "train",
        sequences=cfg.data.train_sequences,
        max_points=cfg.train.max_points,
        return_corr_indices=True,
        matching_radius_2d=cfg.data.matching_radius_2d,
        matching_radius_3d=cfg.data.matching_radius_3d,
        overlap_threshold=cfg.data.overlap_threshold,
        use_augmentation=True,
        image_h=cfg.data.image_h,
        image_w=cfg.data.image_w,
    )

    neighbor_limits = calibrate_neighbors_pack_mode(
        train_dataset,
        GraphPyramid2D3DRegistrationCollateFn,
        cfg.model.point_backbone.num_stages,
        cfg.model.point_backbone.base_voxel_size,
        cfg.model.point_backbone.base_voxel_size * cfg.model.point_backbone.kpconv_radius,
    )

    collate_fn = _make_collate_fn(cfg, neighbor_limits)

    train_loader = build_dataloader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        collate_fn=collate_fn,
    )

    valid_dataset = R3LiveDataset(
        cfg.data.dataset_dir,
        "val",
        sequences=cfg.data.val_sequences,
        max_points=cfg.test.max_points,
        return_corr_indices=True,
        matching_radius_2d=cfg.data.matching_radius_2d,
        matching_radius_3d=cfg.data.matching_radius_3d,
        overlap_threshold=cfg.data.overlap_threshold,
        image_h=cfg.data.image_h,
        image_w=cfg.data.image_w,
    )

    valid_loader = build_dataloader(
        valid_dataset,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return train_loader, valid_loader, neighbor_limits


def test_data_loader(cfg):
    # calibrate neighbor limits using training set
    train_dataset = R3LiveDataset(
        cfg.data.dataset_dir,
        "train",
        sequences=cfg.data.train_sequences,
        max_points=cfg.train.max_points,
        return_corr_indices=True,
        matching_radius_2d=cfg.data.matching_radius_2d,
        matching_radius_3d=cfg.data.matching_radius_3d,
        overlap_threshold=cfg.data.overlap_threshold,
        use_augmentation=True,
        image_h=cfg.data.image_h,
        image_w=cfg.data.image_w,
    )

    neighbor_limits = calibrate_neighbors_pack_mode(
        train_dataset,
        GraphPyramid2D3DRegistrationCollateFn,
        cfg.model.point_backbone.num_stages,
        cfg.model.point_backbone.base_voxel_size,
        cfg.model.point_backbone.base_voxel_size * cfg.model.point_backbone.kpconv_radius,
    )

    collate_fn = _make_collate_fn(cfg, neighbor_limits)

    test_dataset = R3LiveDataset(
        cfg.data.dataset_dir,
        "test",
        sequences=cfg.data.test_sequences,
        max_points=cfg.test.max_points,
        return_corr_indices=True,
        matching_radius_2d=cfg.data.matching_radius_2d,
        matching_radius_3d=cfg.data.matching_radius_3d,
        overlap_threshold=cfg.data.overlap_threshold,
        image_h=cfg.data.image_h,
        image_w=cfg.data.image_w,
    )

    test_loader = build_dataloader(
        test_dataset,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return test_loader, neighbor_limits
