#!/usr/bin/env python3

"""Generates a feature-based object representation."""

import os
import logging

from typing import Any, Dict, List, NamedTuple, Optional

import torch

from utils.misc import array_to_tensor

from bop_toolkit_lib import inout, dataset_params

import bop_toolkit_lib.config as bop_config

from typing import List, Tuple

import kornia
import torch.nn.functional as F

from utils import logging, misc, geometry
from utils.structs import PinholePlaneCameraModel

from utils import dinov2_utils

from utils import (
    cluster_util,
    feature_util,
    projector_util,
    repre_util,
    template_util,
    preprocess_util,
    config_util,
    json_util,
    logging,
    misc,
    structs,
)

from utils.structs import PinholePlaneCameraModel

import numpy as np

from dataclasses import dataclass

@dataclass
class FeatureBasedObjectRepre:
    templates_feature_list: Optional[list] = None
    avg_feature: Optional[torch.Tensor] = None

# def save_object_repre(
#     repre: FeatureBasedObjectRepre,
#     repre_dir: str,
# ) -> None:
#
#     # Save the object into torch data.
#
#     object_dict = {}
#
#     for key, value in repre.__dict__.items():
#         if value is not None and torch.is_tensor(value):
#             object_dict[key] = value
#
#     # Save camera metadata.
#     object_dict["template_cameras_cam_from_model"] = []
#     for camera in repre.template_cameras_cam_from_model:
#         cam_data = {
#             "f": torch.tensor(camera.f),
#             "c": torch.tensor(camera.c),
#             "width": camera.width,
#             "height": camera.height,
#             "T_world_from_eye": torch.tensor(camera.T_world_from_eye),
#         }
#         object_dict["template_cameras_cam_from_model"].append(cam_data)
#
#
#     object_dict["feat_opts"] = repre.feat_opts._asdict()
#     object_dict["template_desc_opts"] = repre.template_desc_opts._asdict()
#
#
#     object_dict["feat_raw_projectors"] = []
#     for projector in repre.feat_raw_projectors:
#         object_dict["feat_raw_projectors"].append(projector_util.projector_to_tensordict(projector))
#
#     object_dict["feat_vis_projectors"] = []
#     for projector in repre.feat_vis_projectors:
#         object_dict["feat_vis_projectors"].append(projector_util.projector_to_tensordict(projector))
#
#     # Save the dictionary of tensors to the file
#     repre_path = os.path.join(repre_dir, "feature_1024.pth")
#
#     torch.save(object_dict, repre_path)
def generate_grid_points(
    grid_size: Tuple[int, int],
    cell_size: float = 1.0,
) -> torch.Tensor:
    """Generates 2D coordinates at the centers of cells of a regular grid.
    Args:
        grid_size: Size of the grid expressed as (grid_width, grid_height).
        cell_size: Size of a cell expressed as a single number as cell are square.
    Returns:
        Generated 2D coordinates.
    """

    # Width and height of the grid expressed in the number of cells.
    grid_cols = int(grid_size[0] / cell_size)
    grid_rows = int(grid_size[1] / cell_size)

    # Generate 2D coordinates at the centers of the grid cells.
    cell_half_size = cell_size / 2.0
    x = torch.linspace(
        cell_half_size, grid_size[0] - cell_half_size, grid_cols, dtype=torch.float
    )
    y = torch.linspace(
        cell_half_size, grid_size[1] - cell_half_size, grid_rows, dtype=torch.float
    )
    grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")

    # 2D coordinates of shape (num_points, 2).
    return torch.vstack((grid_x.flatten(), grid_y.flatten())).T

def filter_points_by_box(
    points: torch.Tensor, box: Tuple[float, float, float, float]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Keeps only points inside the specified 2D bounding box.

    Args:
        points: 2D coordinates of shape (num_points, 2).
        box: A bounding box expressed as (x1, y1, x2, y2).
    Returns:
        Filtered points and the mask.
    """

    x1, y1, x2, y2 = box
    valid_mask = torch.logical_and(
        torch.logical_and(points[:, 0] > x1, points[:, 0] < x2),
        torch.logical_and(points[:, 1] > y1, points[:, 1] < y2),
    )
    return points[valid_mask], valid_mask


def filter_points_by_mask(points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Keeps only points inside the specified mask.

    Args:
        points: 2D coordinates of shape (num_points, 2).
        mask: A binary mask.
    Returns:
        Filtered points.
    """

    # Convert the integers so we can use the coordinates for indexing.
    # Add 0.5 to convert to image coordinates before masking.
    points_int = (points + 0.5).int()

    # Keep only points inside the canvas.
    points_int, valid_mask = filter_points_by_box(
        points_int, (0, 0, mask.shape[1], mask.shape[0])
    )

    # Keep only points inside the mask.
    points = points[valid_mask][mask[points_int[:, 1], points_int[:, 0]].bool()]

    return points
def sample_feature_map_at_points(
    feature_map_chw: torch.Tensor, points: torch.Tensor, image_size: Tuple[int, int]
) -> torch.Tensor:
    """Samples a feature map at the specified 2D coordinates.

    Args:
        feature_map_chw: A tensor of shape (C, H, W).
        points: A tensor of shape (N, 2) where N is the number of points.
        image_size: Size of the input image expressed as (image_width, image_height).
            2D coordinates of the points are expressed in the image coordinates.
    Returns:
        A tensor of shape (num_points, feature_dim) containing the sampled
        features at given 2D coordinates.
    """

    # Normalize the 2D coordinates to [-1, 1].
    uv = torch.div(2.0, torch.as_tensor(image_size)).to(points.device) * points - 1.0

    # Convert the 2D coordinates to shape (1, N, 1, 2).
    query_coords = uv.unsqueeze(0).unsqueeze(2)

    # Feature vectors of shape [1, C, N, 1].
    features = torch.nn.functional.grid_sample(
        feature_map_chw.unsqueeze(0),
        query_coords,
        align_corners=False,
    )

    # Reshape the feature vectors to (N, C).
    features = features[0, :, :, 0].permute(1, 0)

    return features
def lift_2d_points_to_3d(
    points: torch.Tensor,
    depth_image: torch.Tensor,
    camera_model: PinholePlaneCameraModel,
) -> torch.Tensor:
    device = points.device

    # The considered focal length is the average of fx and fy.
    focal = 0.5 * (camera_model.f[0] + camera_model.f[1])

    # 3D points in the camera space.
    points_3d_in_cam = torch.hstack(
        [
            points - torch.as_tensor(camera_model.c).to(torch.float32).to(device),
            focal * torch.ones(points.shape[0], 1).to(torch.float32).to(device),
        ]
    )
    depths = depth_image[
        torch.floor(points[:, 1]).to(torch.int32),
        torch.floor(points[:, 0]).to(torch.int32),
    ].reshape(-1, 1)
    points_3d_in_cam *= depths / points_3d_in_cam[:, 2].reshape(-1, 1)

    return points_3d_in_cam
def get_visual_features_registered_in_3d(
    image_chw: torch.Tensor,
    depth_image_hw: torch.Tensor,
    object_mask: torch.Tensor,
    camera: PinholePlaneCameraModel,
    T_model_from_camera: torch.Tensor,
    extractor: torch.nn.Module,
    grid_cell_size: float,
    debug: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    device = image_chw.device

    timer = misc.Timer(enabled=debug)
    timer.start()

    # Generate grid points at which to sample feature vectors.
    grid_points = generate_grid_points(
        grid_size=(image_chw.shape[2], image_chw.shape[1]),
        cell_size=grid_cell_size,
    ).to(device)

    # Erode the mask a bit to ignore pixels at the contour where
    # depth values tend to be noisy.
    kernel = torch.ones(5, 5).to(device)
    object_mask_eroded = (
        kornia.morphology.erosion(
            object_mask.reshape(1, 1, *object_mask.shape).to(torch.float32), kernel
        )
        .squeeze([0, 1])
        .to(object_mask.dtype)
    )

    # Keep only grid points inside the object mask.
    query_points = filter_points_by_mask(grid_points, object_mask_eroded)

    # Get 3D coordinates corresponding to the query points.
    vertices_in_cam = lift_2d_points_to_3d(
        points=query_points,
        depth_image=depth_image_hw,
        camera_model=camera,
    )
    # Transform vertices to the model space.
    vertices_in_model = geometry.transform_3d_points_torch(
        T_model_from_camera.to(device), vertices_in_cam.to(device)
    )
    vertex_ids = torch.arange(vertices_in_model.shape[0], dtype=torch.int32)

    timer.elapsed("Time for preparation")

    # Extract feature vectors.
    timer.start()
    image_bchw = image_chw.unsqueeze(0)

    timer.start()

    # Extract feature map at the current image scale.
    extractor_output = extractor(image_bchw)
    feature_map_chw = extractor_output["feature_maps"][0]
    feature_cls= extractor_output["cls_tokens"][0].detach()
    feature_map_chw = feature_map_chw.to(device)

    timer.elapsed(f"Time for feature extraction")
    timer.start()

    # Extract feature vectors at query points.
    feat_vectors = sample_feature_map_at_points(
        feature_map_chw=feature_map_chw,
        points=query_points,
        image_size=(image_chw.shape[-1], image_chw.shape[-2]),
    ).detach()

    timer.elapsed(f"Time for feature sampling.")

    return (
        feat_vectors,
        vertex_ids,
        vertices_in_model,
        feature_cls,
    )

class GenRepreOpts(NamedTuple):
    """Options that can be specified via the command line."""

    version: str
    templates_version: str
    object_dataset: str
    object_lids: Optional[List[int]] = None

    # Feature extraction options.
    extractor_name: str = "dinov2_vits14_reg"
    grid_cell_size: float = 14.0

    # Feature PCA options.
    apply_pca: bool = True
    pca_components: int = 256
    pca_whiten: bool = False
    pca_max_samples_for_fitting: int = 100000

    # Feature clustering options.
    cluster_features: bool = True
    cluster_num: int = 2048

    # Template descriptor options.
    template_desc_opts: Optional[repre_util.TemplateDescOpts] = None

    # Other options.
    overwrite: bool = True
    debug: bool = True


def generate_raw_repre(
        opts: GenRepreOpts,
        object_dataset: str,
        object_lid: int,
        extractor: torch.nn.Module,
        output_dir: str,
        device: str = "cuda",
        debug: bool = False,
) -> repre_util.FeatureBasedObjectRepre:
    logger = logging.get_logger(level=logging.INFO if opts.debug else logging.WARNING)

    # Prepare a timer.
    timer = misc.Timer(enabled=debug)

    datasets_path = bop_config.datasets_path

    # Load the template metadata.
    # metadata_path = "/Users/evinpinar/Documents/opensource_foundpose/output/templates/v1/lmo/1/metadata.json"
    metadata_path = os.path.join(
        bop_config.output_path,
        "templates",
        opts.templates_version,
        opts.object_dataset,
        str(object_lid),
        "metadata.json"
    )
    metadata = json_util.load_json(metadata_path)

    # Prepare structures for storing data.
    feat_vectors_list = []
    feat_to_vertex_ids_list = []
    vertices_in_model_list = []
    feat_to_template_ids_list = []
    templates_list = []
    template_cameras_cam_from_model_list = []

    templates_feature_cls = []

    # Use template images specified in the metadata.
    template_id = 0
    num_templates = len(metadata)
    for data_id, data_sample in enumerate(metadata):
        logger.info(
            f"Processing dataset {data_id}/{num_templates}, "
        )

        timer.start()

        camera_sample = data_sample["cameras"]
        camera_world_from_cam = PinholePlaneCameraModel(
            width=camera_sample["ImageSizeX"],
            height=camera_sample["ImageSizeY"],
            f=(camera_sample["fx"], camera_sample["fy"]),
            c=(camera_sample["cx"], camera_sample["cy"]),
            T_world_from_eye=np.array(camera_sample["T_WorldFromCamera"])
        )

        # RGB/monochrome and depth images (in mm).
        image_path = data_sample["rgb_image_path"]
        depth_path = data_sample["depth_map_path"]
        mask_path = data_sample["binary_mask_path"]

        image_arr = inout.load_im(image_path)  # H,W,C
        depth_image_arr = inout.load_depth(depth_path)
        mask_image_arr = inout.load_im(mask_path)

        image_chw = array_to_tensor(image_arr).to(torch.float32).permute(2, 0, 1).to(device) / 255.0
        depth_image_hw = array_to_tensor(depth_image_arr).to(torch.float32).to(device)
        object_mask_modal = array_to_tensor(mask_image_arr).to(torch.float32).to(device)

        # Get the object annotation.
        assert data_sample["dataset"] == object_dataset
        assert data_sample["lid"] == object_lid
        assert data_sample["template_id"] == data_id

        object_pose = data_sample["pose"]

        # Transformations.
        object_pose_rigid_matrix = np.eye(4)
        object_pose_rigid_matrix[:3, :3] = object_pose["R"]
        object_pose_rigid_matrix[:3, 3:] = object_pose["t"]
        T_world_from_model = (
            array_to_tensor(object_pose_rigid_matrix)
            .to(torch.float32)
            .to(device)
        )
        T_model_from_world = torch.linalg.inv(T_world_from_model)
        T_world_from_camera = (
            array_to_tensor(camera_world_from_cam.T_world_from_eye)
            .to(torch.float32)
            .to(device)
        )
        T_model_from_camera = torch.matmul(T_model_from_world, T_world_from_camera)

        timer.elapsed("Time for getting template data")
        timer.start()

        # Extract features from the current template.
        (
            feat_vectors,
            feat_to_vertex_ids,
            vertices_in_model,
            feature_cls,
        ) = get_visual_features_registered_in_3d(
            image_chw=image_chw,
            depth_image_hw=depth_image_hw,
            object_mask=object_mask_modal,
            camera=camera_world_from_cam,
            T_model_from_camera=T_model_from_camera,
            extractor=extractor,
            grid_cell_size=opts.grid_cell_size,
            debug=False,
        )
        templates_feature_cls.append(feature_cls)


        timer.elapsed("Time for feature extraction")
        timer.start()

        # Store data.
        feat_vectors_list.append(feat_vectors)
        feat_to_vertex_ids_list.append(feat_to_vertex_ids)
        vertices_in_model_list.append(vertices_in_model)
        feat_to_template_ids = template_id * torch.ones(
            feat_vectors.shape[0], dtype=torch.int32, device=device
        )
        feat_to_template_ids_list.append(feat_to_template_ids)

        # Save the template as uint8 to save space.
        image_chw_uint8 = (image_chw * 255).to(torch.uint8)
        templates_list.append(image_chw_uint8)

        # Store camera model of the current template.
        camera_model = camera_world_from_cam.copy()
        camera_model.extrinsics = torch.linalg.inv(T_model_from_camera)
        template_cameras_cam_from_model_list.append(camera_model)

        # Increment the template ID.
        template_id += 1

        timer.elapsed("Time for storing data")

    logger.info("Processing done.")

    all_feature_vector = feat_vectors_list
    avg_feature_vector = [torch.mean(it,dim=0) for it in feat_vectors_list] # len()=798, element.shape=1024
    avg_feature_vector = torch.stack(avg_feature_vector)
    templates_feature_cls = torch.stack(templates_feature_cls)
    # Build the object representation from the collected data.
    # return FeatureBasedObjectRepre(
    #     templates_feature_list =  all_feature_vector, # list, len = 798, elements is query feature vector, shape = 170+,1024
    #     avg_feature = avg_feature_vector, # shape = (798,1024)
    # )

    return {
        "templates_feature_list":all_feature_vector,
        "avg_feature":avg_feature_vector,
        "cls_token":templates_feature_cls
    }




def generate_repre(
        opts: GenRepreOpts,
        dataset: str,
        lid: int,
        device: str = "cuda",
        extractor: Optional[torch.nn.Module] = None,
) -> None:
    logger = logging.get_logger(level=logging.INFO if opts.debug else logging.WARNING)

    datasets_path = bop_config.datasets_path

    # Prepare a timer.
    timer = misc.Timer(enabled=opts.debug)
    timer.start()

    # Prepare the output folder.
    base_repre_dir = os.path.join(bop_config.output_path, "object_repre")
    output_dir = repre_util.get_object_repre_dir_path(
        base_repre_dir, opts.version, dataset, lid
    )
    if os.path.exists(output_dir) and not opts.overwrite:
        raise ValueError(f"Output directory already exists: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Save parameters to a JSON file.
    json_util.save_json(os.path.join(output_dir, "config.json"), opts)

    # Prepare a feature extractor.
    if extractor is None:
        extractor = feature_util.make_feature_extractor(opts.extractor_name)
    extractor.to(device)

    timer.elapsed("Time for preparation")
    timer.start()

    # Build raw object representation.
    features_1024 = generate_raw_repre(
        opts=opts,
        object_dataset=dataset,
        object_lid=lid,
        extractor=extractor,
        output_dir=output_dir,
        device=device,
    )
    # metadata_path = os.path.join(
    #     bop_config.output_path,
    #     "templates",
    #     opts.templates_version,
    #     opts.object_dataset,
    #     str(lid),
    #     "metadata.json"
    # )
    # metadata = json_util.load_json(metadata_path)
    # cls_list= []
    # for data_id, data_sample in enumerate(metadata):
    #     image_path = data_sample["rgb_image_path"]
    #     image_arr = inout.load_im(image_path)  # H,W,C
    #     image_chw = array_to_tensor(image_arr).to(torch.float32).permute(2, 0, 1).to(device) / 255.0
    #     image_bchw = image_chw.unsqueeze(0)
    #     extractor_output = extractor(image_bchw)
    #     cls_list.append(extractor_output["cls_tokens"][0])
    #
    # features_cls_1024 = torch.stack(cls_list) # 798,1024
    repre_dir = repre_util.get_object_repre_dir_path(
        base_repre_dir, opts.version, dataset, lid
    )
    torch.save(features_1024, repre_dir+ "/features_1024.pth")
    print("finish" + str(lid))



def generate_repre_from_list(opts: GenRepreOpts) -> None:
    # Get IDs of objects to process.
    object_lids = opts.object_lids
    if object_lids is None:
        datasets_path = bop_config.datasets_path
        bop_model_props = dataset_params.get_model_params(datasets_path=datasets_path, dataset_name=opts.object_dataset)
        object_lids = bop_model_props["obj_ids"]

    # Prepare a feature extractor.
    extractor = feature_util.make_feature_extractor(opts.extractor_name)

    # Prepare a device.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: ", device)

    # Process each image separately.
    for object_lid in object_lids:
        generate_repre(opts, opts.object_dataset, object_lid, device, extractor)


def main() -> None:
    main_config = config_util.load_opts_from_json_or_command_line(GenRepreOpts)[0]
    generate_repre_from_list(main_config)


if __name__ == "__main__":
    main()
