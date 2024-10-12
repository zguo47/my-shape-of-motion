import argparse
import fnmatch
import os
import os.path as osp
from glob import glob
from typing import Literal

import cv2
import imageio.v2 as iio
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Pipeline, pipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UINT16_MAX = 65535


models = {
    "depth-anything": "LiheYoung/depth-anything-large-hf",
    "depth-anything-v2": "depth-anything/Depth-Anything-V2-Large-hf",
}


def get_pipeline(model_name: str):
    pipe = pipeline(task="depth-estimation", model=models[model_name], device=DEVICE)
    print(f"{model_name} model loaded.")
    return pipe


def to_uint16(disp: np.ndarray):
    disp_min = disp.min()
    disp_max = disp.max()

    if disp_max - disp_min > np.finfo("float").eps:
        disp_uint16 = UINT16_MAX * (disp - disp_min) / (disp_max - disp_min)
    else:
        disp_uint16 = np.zeros(disp.shape, dtype=disp.dtype)
    disp_uint16 = disp_uint16.astype(np.uint16)
    return disp_uint16


def get_depth_anything_disp(
    pipe: Pipeline,
    img_file: str,
    ret_type: Literal["uint16", "float"] = "float",
):

    image = Image.open(img_file)
    disp = pipe(image)["predicted_depth"]
    disp = torch.nn.functional.interpolate(
        disp.unsqueeze(1), size=image.size[::-1], mode="bicubic", align_corners=False
    )
    disp = disp.squeeze().cpu().numpy()
    if ret_type == "uint16":
        return to_uint16(disp)
    elif ret_type == "float":
        return disp
    else:
        raise ValueError(f"Unknown return type {ret_type}")


def save_disp_from_dir(
    model_name: str,
    img_dir: str,
    out_dir: str,
    matching_pattern: str = "*",
):
    img_files = sorted(glob(osp.join(img_dir, "*.jpg"))) + sorted(
        glob(osp.join(img_dir, "*.png"))
    )
    img_files = [
        f for f in img_files if fnmatch.fnmatch(osp.basename(f), matching_pattern)
    ]
    if osp.exists(out_dir) and len(glob(osp.join(out_dir, "*.png"))) == len(img_files):
        print(f"Raw {model_name} depth maps already computed for {img_dir}")
        return

    pipe = get_pipeline(model_name)
    os.makedirs(out_dir, exist_ok=True)
    for img_file in tqdm(img_files, f"computing {model_name} depth maps"):
        disp = get_depth_anything_disp(pipe, img_file, ret_type="uint16")
        out_file = osp.join(out_dir, osp.splitext(osp.basename(img_file))[0] + ".png")
        iio.imwrite(out_file, disp)


def align_monodepth_with_metric_depth(
    metric_depth_dir: str,
    monodepth_npz_path: str,
    output_monodepth_dir: str,
    matching_pattern: str = "*",
):
    print(
        f"Aligning monodepth from {monodepth_npz_path} with metric depth in {metric_depth_dir}"
    )

    monodepth_data = np.load(monodepth_npz_path)
    mono_depth_array = monodepth_data['depth']  # Shape: (N, 250, 512)

    new_height = 480
    new_width = 854
    resized_depth_maps = []
    for depth_map in mono_depth_array:
        # Resize depth_map from (250, 512) to (480, 854)
        resized_map = cv2.resize(
            depth_map,
            (new_width, new_height),
            interpolation=cv2.INTER_LINEAR
        )
        resized_depth_maps.append(resized_map)
    mono_depth_array_resized = np.array(resized_depth_maps)  # Shape: (N, 480, 854)

    metric_paths = sorted(glob(f"{metric_depth_dir}/{matching_pattern}"))
    img_files = [osp.basename(p) for p in metric_paths]
    img_files.sort()

    os.makedirs(output_monodepth_dir, exist_ok=True)
    # if len(os.listdir(output_monodepth_dir)) == len(img_files):
    #     print(f"Found {len(img_files)} files in {output_monodepth_dir}, skipping")
    #     return

    if len(img_files) != mono_depth_array_resized.shape[0]:
        raise ValueError(
            "Number of images does not match the number of depth maps in the monodepth npz file."
        )

    frame_idx = 0  # Using frame 0

    imname = osp.splitext(img_files[frame_idx])[0]
    metric_path = osp.join(metric_depth_dir, imname + ".npy")

    mono_disp_map = mono_depth_array_resized[frame_idx]
    mono_disp_map = mono_disp_map / UINT16_MAX  

    metric_disp_map = np.load(metric_path)
    ms_colmap_disp = metric_disp_map - np.median(metric_disp_map) + 1e-8
    ms_mono_disp = mono_disp_map - np.median(mono_disp_map) + 1e-8

    scale = np.median(ms_colmap_disp / ms_mono_disp)
    shift = np.median(metric_disp_map - scale * mono_disp_map)

    print(f"Global scale (from frame 0): {scale}")
    print(f"Global shift (from frame 0): {shift}")

    for idx, f in enumerate(tqdm(img_files, desc="Applying global scale and shift")):
        imname = osp.splitext(f)[0]
        mono_disp_map = mono_depth_array_resized[idx]
        mono_disp_map = mono_disp_map / UINT16_MAX  

        aligned_disp = scale * mono_disp_map + shift

        # Set depth values that are too small to invalid (0)
        min_thre = min(1e-6, np.quantile(aligned_disp, 0.01))
        aligned_disp[aligned_disp < min_thre] = 0.0

        out_file = osp.join(output_monodepth_dir, imname + ".npy")
        np.save(out_file, aligned_disp)


def align_monodepth_with_colmap(
    sparse_dir: str,
    input_monodepth_dir: str,
    output_monodepth_dir: str,
    matching_pattern: str = "*",
):
    from pycolmap import SceneManager

    manager = SceneManager(sparse_dir)
    manager.load()

    cameras = manager.cameras
    images = manager.images
    points3D = manager.points3D
    point3D_id_to_point3D_idx = manager.point3D_id_to_point3D_idx

    bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
    os.makedirs(output_monodepth_dir, exist_ok=True)
    images = [
        image
        for _, image in images.items()
        if fnmatch.fnmatch(image.name, matching_pattern)
    ]

    monodepth_data = np.load(input_monodepth_dir)

    for image in tqdm(images, desc="Aligning monodepth with colmap point cloud"):

        point3D_ids = image.point3D_ids
        point3D_ids = point3D_ids[point3D_ids != manager.INVALID_POINT3D]
        pts3d_valid = points3D[
            [point3D_id_to_point3D_idx[id] for id in point3D_ids]
        ]  # type: ignore
        K = cameras[image.camera_id].get_camera_matrix()
        rot = image.R()
        trans = image.tvec.reshape(3, 1)
        extrinsics = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)

        pts3d_valid_homo = np.concatenate(
            [pts3d_valid, np.ones_like(pts3d_valid[..., :1])], axis=-1
        )
        pts3d_valid_cam_homo = extrinsics.dot(pts3d_valid_homo.T).T
        pts2d_valid_cam = K.dot(pts3d_valid_cam_homo[..., :3].T).T
        pts2d_valid_cam = pts2d_valid_cam[..., :2] / pts2d_valid_cam[..., 2:3]
        colmap_depth = pts3d_valid_cam_homo[..., 2]

        image_base_name = osp.splitext(image.name)[0]
        if image_base_name in monodepth_data:
            mono_disp_map = monodepth_data[image_base_name]
        else:
            print(f"Monodepth data for image {image_base_name} not found.")
            continue

        mono_disp_map = mono_disp_map / UINT16_MAX

        colmap_disp = 1.0 / np.clip(colmap_depth, a_min=1e-6, a_max=1e6)
        mono_disp = cv2.remap(
            mono_disp_map,  # type: ignore
            pts2d_valid_cam[None, ...].astype(np.float32),
            None,  # type: ignore
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )[0]
        ms_colmap_disp = colmap_disp - np.median(colmap_disp) + 1e-8
        ms_mono_disp = mono_disp - np.median(mono_disp) + 1e-8

        scale = np.median(ms_colmap_disp / ms_mono_disp)
        shift = np.median(colmap_disp - scale * mono_disp)

        mono_disp_aligned = scale * mono_disp_map + shift

        min_thre = min(1e-6, np.quantile(mono_disp_aligned, 0.01))
        # Set depth values that are too small to invalid (0)
        mono_disp_aligned[mono_disp_aligned < min_thre] = 0.0
        np.save(
            osp.join(output_monodepth_dir, image_base_name + ".npy"),
            mono_disp_aligned,
        )


def main():
    align_monodepth_with_metric_depth(
    metric_depth_dir="/fs/nexus-projects/video-depth-pose/videosfm/shape-of-motion/davis/unidepth_disp/480p/hike",
    monodepth_npz_path="/fs/nexus-projects/video-depth-pose/videosfm/shape-of-motion/davis/depthcrafter/hike/hike30.npz",
    output_monodepth_dir="/fs/nexus-projects/video-depth-pose/videosfm/shape-of-motion/davis/aligned_depthcrafter/480p/hike",
    matching_pattern="*.npy"
)



if __name__ == "__main__":
    """ example usage for iphone dataset:
    python compute_depth.py \
        --img_dir /home/qianqianwang_google_com/datasets/iphone/dycheck/paper-windmill/rgb/1x \
        --out_raw_dir /home/qianqianwang_google_com/datasets/iphone/dycheck/paper-windmill/flow3d_preprocessed/depth_anything_v2/1x \
        --out_aligned_dir /home/qianqianwang_google_com/datasets/iphone/dycheck/paper-windmill/flow3d_preprocessed/aligned_depth_anything_v2/1x \
        --sparse_dir /home/qianqianwang_google_com/datasets/iphone/dycheck/paper-windmill/flow3d_preprocessed/colmap/sparse \
        --matching_pattern "0_*"
    """
    main()