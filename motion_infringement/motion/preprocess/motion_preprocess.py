"""
motion_preprocess.py
--------------------
Generated (suspected) video -> SMPL motion (.npz), same format as the DB.

Pipeline
  1) read video frames (OpenCV)
  2) person detection (torchvision Faster R-CNN), keep one tracked person
  3) HybrIK (HRNet-W48) -> SMPL 24 joint rotations per frame
  4) rotation matrix -> axis-angle, resample to 30 fps (DB frame rate)
  5) save npz: poses_axis_angle [T, 24, 3], trans [T, 3]

Usage
  python motion_preprocess.py \
      --video ../../../suspect_data/a.mp4 \
      --out   ../../../results/a/motion/a_smpl.npz \
      --hybrik_root /path/to/HybrIK
"""
import argparse
import os
import sys
import warnings

import cv2
import numpy as np
import torch
from easydict import EasyDict as edict
from scipy.spatial.transform import Rotation as R, Slerp
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tqdm import tqdm

warnings.filterwarnings("ignore")

TARGET_FPS = 30.0  # DB motion sequences are 30 fps
DEFAULT_HYBRIK = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              "..", "..", "..", "third_party", "HybrIK"))

# chumpy (used to unpickle the SMPL model) imports numpy aliases removed in numpy>=1.24
for _n, _t in [("bool", bool), ("int", int), ("float", float), ("complex", complex),
               ("object", object), ("unicode", str), ("str", str)]:
    if not hasattr(np, _n):
        setattr(np, _n, _t)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--hybrik_root", default=DEFAULT_HYBRIK,
                   help="HybrIK repo root (configs/, model_files/, pretrained_models/)")
    p.add_argument("--ckpt", default="pretrained_models/hybrik_hrnet.pth")
    p.add_argument("--cfg", default="configs/256x192_adam_lr1e-3-hrw48_cam_2x_w_pw3d_3dhp.yaml")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--target_fps", type=float, default=TARGET_FPS)
    return p.parse_args()


def build_models(args):
    root = os.path.abspath(args.hybrik_root)
    sys.path.insert(0, root)
    ckpt = os.path.abspath(args.ckpt) if os.path.exists(args.ckpt) else os.path.join(root, args.ckpt)

    cwd = os.getcwd()
    os.chdir(root)  # HybrIK uses relative model_files/ paths at import time
    try:
        from hybrik.models import builder
        from hybrik.utils.config import update_config
        from hybrik.utils.presets import SimpleTransform3DSMPLCam

        cfg = update_config(args.cfg)
        bbox_3d_shape = [x * 1e-3 for x in getattr(cfg.MODEL, "BBOX_3D_SHAPE", (2000, 2000, 2000))]
        dummy_set = edict({"joint_pairs_17": None, "joint_pairs_24": None,
                           "joint_pairs_29": None, "bbox_3d_shape": bbox_3d_shape})
        transform = SimpleTransform3DSMPLCam(
            dummy_set, scale_factor=cfg.DATASET.SCALE_FACTOR,
            color_factor=cfg.DATASET.COLOR_FACTOR, occlusion=cfg.DATASET.OCCLUSION,
            input_size=cfg.MODEL.IMAGE_SIZE, output_size=cfg.MODEL.HEATMAP_SIZE,
            depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM, bbox_3d_shape=bbox_3d_shape,
            rot=cfg.DATASET.ROT_FACTOR, sigma=cfg.MODEL.EXTRA.SIGMA,
            train=False, add_dpg=False, loss_type=cfg.LOSS["TYPE"])

        model = builder.build_sppe(cfg.MODEL)
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    finally:
        os.chdir(cwd)

    det = fasterrcnn_resnet50_fpn(pretrained=True)
    dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    det.to(dev).eval()
    model.to(dev).eval()
    return det, model, transform, dev


def resample_poses(poses, trans, src_fps, dst_fps):
    """Resample axis-angle poses with per-joint SLERP, trans with linear interp."""
    T_src = len(poses)
    if T_src < 2 or abs(src_fps - dst_fps) < 1e-3:
        return poses, trans
    t_src = np.arange(T_src) / src_fps
    t_dst = np.arange(0, t_src[-1] + 1e-9, 1.0 / dst_fps)
    out = np.zeros((len(t_dst), poses.shape[1], 3), dtype=np.float32)
    for j in range(poses.shape[1]):
        out[:, j] = Slerp(t_src, R.from_rotvec(poses[:, j]))(t_dst).as_rotvec()
    trans_out = np.stack([np.interp(t_dst, t_src, trans[:, k]) for k in range(3)], axis=1)
    return out, trans_out.astype(np.float32)


def run(args):
    args.video = os.path.abspath(args.video)
    args.out = os.path.abspath(args.out)
    det, model, transform, dev = build_models(args)
    from hybrik.utils.vis import get_max_iou_box, get_one_box

    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), f"Cannot open video: {args.video}"
    fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    to_tensor = T.Compose([T.ToTensor()])

    poses, transl, prev_box = [], [], None
    last_pose, last_transl = None, None
    n_missing = 0

    for _ in tqdm(range(n_frames), desc="HybrIK"):
        ok, frame = cap.read()
        if not ok:
            break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            det_out = det([to_tensor(img).to(dev)])[0]
            box = get_one_box(det_out) if prev_box is None else get_max_iou_box(det_out, prev_box)

            if box is None:  # no person: repeat last pose to keep timeline
                n_missing += 1
                if last_pose is not None:
                    poses.append(last_pose)
                    transl.append(last_transl)
                continue
            prev_box = box

            inp, bbox, img_center = transform.test_transform(img, box)
            out = model(
                inp.to(dev)[None], flip_test=True,
                bboxes=torch.tensor(np.array(bbox), device=dev).unsqueeze(0).float(),
                img_center=torch.from_numpy(img_center).to(dev).unsqueeze(0).float())

            rotmat = out.pred_theta_mats.reshape(24, 3, 3).cpu().numpy()
            last_pose = R.from_matrix(rotmat).as_rotvec().astype(np.float32)
            last_transl = out.transl.reshape(3).cpu().numpy().astype(np.float32)
            poses.append(last_pose)
            transl.append(last_transl)
    cap.release()

    assert len(poses) > 1, "No person detected in the video."
    n_read = len(poses) + (n_missing if last_pose is None else 0)
    poses = np.stack(poses)
    transl = np.stack(transl)
    poses, transl = resample_poses(poses, transl, fps, args.target_fps)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, poses_axis_angle=poses.astype(np.float32),
             trans=transl.astype(np.float32), fps=np.float32(args.target_fps),
             src_video=os.path.basename(args.video), src_fps=np.float32(fps))
    print(f"[motion_preprocess] {args.video}: {n_read} frames @ {fps:.2f} fps "
          f"(no-person frames: {n_missing}) -> {poses.shape[0]} frames @ {args.target_fps} fps")
    print(f"[motion_preprocess] saved: {args.out}")


if __name__ == "__main__":
    run(parse_args())
