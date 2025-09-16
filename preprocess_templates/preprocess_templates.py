#!/usr/bin/env python3

"""Infers pose from objects."""

import datetime

import os
import gc
import time

from typing import List, NamedTuple, Optional, Tuple

import cv2

import numpy as np

import torch

from utils.misc import array_to_tensor, tensor_to_array, tensors_to_arrays

from bop_toolkit_lib import inout, dataset_params
import bop_toolkit_lib.config as bop_config
import bop_toolkit_lib.misc as bop_misc


from utils import (
    corresp_util,
    config_util,
    eval_errors,
    eval_util,
    feature_util,
    infer_pose_util,
    knn_util,
    misc as misc_util,
    pnp_util,
    projector_util,
    repre_util,
    vis_util,
    data_util,
    renderer_builder,
    json_util,
    logging,
    misc,
    structs,
)

from utils.structs import AlignedBox2f, PinholePlaneCameraModel
from utils.misc import warp_depth_image, warp_image


logger: logging.Logger = logging.get_logger()


class InferOpts(NamedTuple):
    """Options that can be specified via the command line."""

    version: str
    repre_version: str
    object_dataset: str
    object_lids: Optional[List[int]] = None
    max_sym_disc_step: float = 0.01

    # Cropping options.
    crop: bool = True
    crop_rel_pad: float = 0.2
    crop_size: Tuple[int, int] = (420, 420)

    # Object instance options.
    use_detections: bool = True
    num_preds_factor: float = 1.0
    min_visibility: float = 0.1

    # Feature extraction options.
    extractor_name: str = "dinov2_vitl14"
    grid_cell_size: float = 1.0
    max_num_queries: int = 1000000

    # Feature matching options.
    match_template_type: str = "tfidf"
    match_top_n_templates: int = 5
    match_feat_matching_type: str = "cyclic_buddies"
    match_top_k_buddies: int = 300

    # PnP options.
    pnp_type: str = "opencv"
    pnp_ransac_iter: int = 1000
    pnp_required_ransac_conf: float = 0.99
    pnp_inlier_thresh: float = 10.0
    pnp_refine_lm: bool = True

    final_pose_type: str = "best_coarse"

    # Other options.
    save_estimates: bool = True
    vis_results: bool = True
    vis_corresp_top_n: int = 100
    vis_feat_map: bool = True
    vis_for_paper: bool = True
    debug: bool = True


def infer(opts: InferOpts) -> None:

    datasets_path = bop_config.datasets_path

    # Prepare a logger and a timer.
    logger = logging.get_logger(level=logging.INFO if opts.debug else logging.WARNING)
    timer = misc_util.Timer(enabled=opts.debug)
    timer.start()

    # Load pre-generated detections saved in the BOP format.
    detections = {}
    if opts.use_detections:
        path = os.path.join(
            datasets_path,
            "detections",
            "cnos-fastsam",
            f"cnos-fastsam_{opts.object_dataset}-test.json",
        )
        detections = infer_pose_util.load_detections_in_bop_format(path)


    # Prepare feature extractor.
    extractor = feature_util.make_feature_extractor(opts.extractor_name)
    # Prepare a device.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor.to(device)

    # Get IDs of objects to process.
    object_lids = opts.object_lids # [1,5,6,8,9,10,11,12]
    bop_model_props = dataset_params.get_model_params(datasets_path=datasets_path, dataset_name=opts.object_dataset)
    if object_lids is None:
        # If local (object) IDs are not specified, synthesize templates for all objects
        # in the specified dataset.
        object_lids = bop_model_props["obj_ids"]

    # Get properties of the test split of the specified dataset.
    # 获取数据集基本信息
    bop_test_split_props = dataset_params.get_split_params(
        datasets_path=datasets_path,
        dataset_name=opts.object_dataset,
        split="test"
    )

    # Load BOP test targets
    test_targets_path = os.path.join(bop_test_split_props["base_path"], "test_targets_bop19.json")
    # 加载bop.json中的数据，总计1445项，每一项表示一个实例。
    targets = inout.load_json(test_targets_path)

    scene_ids = dataset_params.get_present_scene_ids(bop_test_split_props)

    scene_im_ids = {}
    '''
    len(test_target_count)=1445, the element such as (2,3,1):1, (2,3,5):1, (2,3,6):1, ...
    len(targets_per_obj) = 8, 依照目标编号，将target中1445个实例划分为8类，并以字典的形式存储。
    '''
    test_target_count = {}
    targets_per_obj = {}
    for target in targets:
        scene_im_ids.setdefault(target["scene_id"], set()).add(target["im_id"])
        key = (target["scene_id"], target["im_id"], target["obj_id"])
        test_target_count[key] = target["inst_count"]
        targets_per_obj.setdefault(target["obj_id"], list()).append(target)

    '''
    scene_gts: {"scene_id":{"img_id":{"obj_id":6d_pose_gt},...},...}
    scene_cameras: {"scene_id":{"img_id":相机内参},...},...}
    scene_gts_info: gt中与6D pose无关的其他信息
    '''
    scene_gts = {}
    scene_gts_info = {}
    scene_cameras = {}
    for scene_id in scene_im_ids.keys():
        scene_cameras[scene_id] = data_util.load_chunk_cameras(bop_test_split_props["scene_camera_tpath"].format(scene_id=scene_id), bop_test_split_props["im_size"])
        scene_gts[scene_id] = data_util.load_chunk_gts(bop_test_split_props["scene_gt_tpath"].format(scene_id=scene_id),opts.object_dataset)
        scene_gts_info[scene_id] = json_util.load_json(
            bop_test_split_props["scene_gt_info_tpath"].format(scene_id=scene_id),
            keys_to_int=True,
        )

    # Create a renderer.
    renderer_type = renderer_builder.RendererType.PYRENDER_RASTERIZER
    renderer = renderer_builder.build(renderer_type=renderer_type, model_path=bop_model_props["model_tpath"])


    timer.elapsed("Time for setting up the stage")

    # Run inference for each specified object.
    obj2subimage_size = {}
    for object_lid in object_lids:
        timer.start()

        # The output folder is named with slugified dataset path.
        version = opts.version
        if version == "":
            version = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        signature = misc.slugify(opts.object_dataset) + "_{}".format(version)
        output_dir = os.path.join(
            bop_config.output_path, "inference", signature, str(object_lid)
        )
        os.makedirs(output_dir, exist_ok=True)

        # Save parameters to a file.
        config_path = os.path.join(output_dir, "config.json")
        json_util.save_json(config_path, opts)

        # Create a pose evaluator.
        pose_evaluator = eval_util.EvaluatorPose([object_lid])

        # Load the object representation.
        logger.info(
            f"Loading representation for object {object_lid} from dataset {opts.object_dataset}..."
        )
        base_repre_dir = os.path.join(bop_config.output_path, "object_repre")
        repre_dir = repre_util.get_object_repre_dir_path(
            base_repre_dir, opts.version, opts.object_dataset, object_lid
        )
        # repre is results by gen_repre.py
        repre = repre_util.load_object_repre(
            repre_dir=repre_dir,
            tensor_device=device,
        )

        logger.info("Object representation loaded.")
        repre_np = repre_util.convert_object_repre_to_numpy(repre)

        # Build a kNN index from object feature vectors.
        visual_words_knn_index = None
        if opts.match_template_type == "tfidf":
            visual_words_knn_index = knn_util.KNN(
                k=repre.template_desc_opts.tfidf_knn_k,
                metric=repre.template_desc_opts.tfidf_knn_metric
            )
            visual_words_knn_index.fit(repre.feat_cluster_centroids) #词向量（聚类中心，形状为2048*256）存入对象中

        # Build per-template KNN index with features from that template.
        '''
        len(template_knn_indices) = 798
        template_knn_indice中的每一个元素代表一个模板，存储其对应query_points的描述符
        '''
        template_knn_indices = []
        if opts.match_feat_matching_type == "cyclic_buddies":
            logger.info("Building per-template KNN indices...")
            for template_id in range(len(repre.template_cameras_cam_from_model)):
                logger.info(f"Building KNN index for template {template_id}...")
                tpl_feat_mask = repre.feat_to_template_ids == template_id
                tpl_feat_ids = torch.nonzero(tpl_feat_mask).flatten()

                template_feats = repre.feat_vectors[tpl_feat_ids]

                # Build knn index for object features.
                template_knn_index = knn_util.KNN(k=1, metric="l2")
                template_knn_index.fit(template_feats.cpu())
                template_knn_indices.append(template_knn_index)
            logger.info("Per-template KNN indices built.")

        logging.log_heading(
            logger,
            f"Object: {object_lid}, vertices: {len(repre.vertices)}",
            style=logging.WHITE_BOLD,
        )

        # Get the object mesh and meta information.
        model_path = bop_model_props["model_tpath"].format(obj_id=object_lid)
        object_mesh = inout.load_ply(model_path)
        models_info = inout.load_json(bop_model_props["models_info_path"], keys_to_int=True)
        object_syms = bop_misc.get_symmetry_transformations(
            models_info[object_lid], max_sym_disc_step= 0.01
        )
        object_diameter = models_info[object_lid]["diameter"]

        max_vertices = 1000
        subsampled_vertices = np.random.permutation(object_mesh["pts"])[:max_vertices]

        timer.elapsed("Time for preparing object data")

        # Perform inference on each selected image.
        # len(targets_per_obj) = 8
        subimage_side_list = []
        for item_info in targets_per_obj[object_lid]: #todo: 遍历编号为obj_id的目标的全部实例
        # for scene_id in scene_ids:
            timer.start()
            # Chunk and image IDs in the original BOP dataset.
            bop_im_id = item_info["im_id"]
            bop_chunk_id = item_info["scene_id"]

            # if bop_chunk_id != 2 or bop_im_id != 322:
            #     continue

            # Get instance identifier if specified.
            inst_id = None
            if "inst_id" in item_info:
                inst_id = item_info["inst_id"]

            sample = data_util.prepare_sample(
                item_info,
                bop_test_split_props,
                scene_cameras,
                scene_gts,
                scene_gts_info
            )

            # Get object annotations.
            object_annos = []
            if sample.objects_anno is not None:
                all_object_annos = None
                if inst_id is not None:
                    all_object_annos = [sample.objects_anno[inst_id]]
                else:
                    all_object_annos = sample.objects_anno

                # Keep only GT annotations for sufficiently visible objects. 对于足够可见的对象，只保留GT注释。
                '''
                all_object_annos: 图像中全部目标的mask标注
                接下来的for循环，目的是根据obj_id获取指定目标的mask标注
                '''
                for anno in all_object_annos:
                    if (
                        anno.lid == object_lid
                        and not np.isnan(anno.visibilities)
                        and anno.visibilities > opts.min_visibility
                    ):
                        object_annos.append(anno)
                        # np.save('gt_mask', anno.masks_modal)

                # Continue if there are no sufficiently visible object annos.
                if len(object_annos) == 0:
                    continue

            # If test targets are specified use them to get the number of target instances.
            # 如果指定了测试目标，则使用它们来获取目标实例的数量。
            sample_key = (bop_chunk_id, bop_im_id, object_lid) # 使用3个索引来定位实例
            if test_target_count is not None:
                # For test images.
                if sample_key not in test_target_count:
                    continue
                # Number of target instances
                num_target_insts = test_target_count[sample_key]

            else:
                num_target_insts = len(object_annos)

            # Skip this image if there are no test instances of the current object.
            if num_target_insts == 0:
                logger.info(f"Skipping image {bop_chunk_id}/{bop_im_id} because no GT.")
                continue

            msg = (
                f"Estimating pose of object {object_lid} in "
                f"scene_id {bop_chunk_id}, im_id {bop_im_id}"
                f"dataset {datasets_path}"
            )
            logging.log_heading(logger, msg, style=logging.BLUE_BOLD)

            # Camera parameters.
            orig_camera_c2w = sample.camera
            orig_image_size = (
                orig_camera_c2w.width,
                orig_camera_c2w.height,
            )

            # Get info about object instances for which we want to estimate pose.
            #  定义待估计实例的对象。
            instances = infer_pose_util.get_instances_for_pose_estimation(
                bop_chunk_id=bop_chunk_id,
                bop_im_id=bop_im_id,
                obj_id=object_lid,
                use_detections=opts.use_detections,
                detections=detections,
                max_num_preds=int(opts.num_preds_factor * num_target_insts),
                gt_object_annos=object_annos, # gt_object_annos, means mask_gt, 从何而来?
                image_size=orig_image_size,
            )
            if len(instances) == 0:
                logger.info("No object instance, skipping.")
                continue

            # Generate grid points at which to sample the feature vectors. 生成用于采样特征向量的网格点。
            if opts.crop:
                grid_size = opts.crop_size
            else:
                grid_size = orig_image_size
            grid_points = feature_util.generate_grid_points(
                grid_size=grid_size,
                cell_size=opts.grid_cell_size,
            )
            grid_points = grid_points.to(device) # grid_points.shape=(900,2)

            timer.elapsed("Time for preparing image data")

            # Estimate pose for each object instance.
            # todo: 为每一个实例估计姿态
            for inst_j, instance in enumerate(instances):
                times = {}

                if opts.use_detections:
                    # Add time for CNOS prediction.
                    pose_evaluator.detection_times[(bop_chunk_id, bop_im_id)] = (
                        instance["time"]
                    )
                    cnos_time = instance["time"]
                    logger.info(f"Time for segmentation: {cnos_time:.5f}s")

                    # Skip the prediction mask if it doesn't overlap with the ground truth.
                    '''
                    instance["input_mask_modal"]: CNOS为实例预测的掩码
                    instance["gt_anno"]: 读取mask_visib目录下的图像构造的真实掩码
                    '''
                    if instance["gt_anno"] is not None:
                        mask_iou = eval_errors.mask_iou(
                            instance["input_mask_modal"],
                            instance["gt_anno"].masks_modal,
                        )

                        if mask_iou < 0.05: #当分割预测结果与mask_GT的IoU小于0.05时，直接跳过该实例
                            continue
                    if (
                        instance["input_mask_modal"].sum()
                        > orig_image_size[0] * orig_image_size[1]
                    ):
                        continue

                    orig_image_np_hwc = sample.image.astype(np.float32) / 255.0

                    # Get the modal mask and amodal bounding box of the instance.
                    orig_mask_modal = instance["input_mask_modal"]
                    mask_bbox = (instance["input_box_amodal"][0],
                    instance["input_box_amodal"][1],
                    instance["input_box_amodal"][2],
                    instance["input_box_amodal"][3],
                    )
                    box_width = mask_bbox[2] - mask_bbox[0]
                    box_high = mask_bbox[3] - mask_bbox[1]
                    # box_center = (mask_bbox[0] + box_width*0.5,mask_bbox[1]+box_high*0.5)
                    box_side = max(box_high,box_width) * 1.8
                    # a = int(box_center[0] - box_side * 0.5)
                    # b = int(box_center[1] - box_side * 0.5)
                    # c = int(box_center[0] + box_side * 0.5)
                    # d = int(box_center[1] + box_side * 0.5)
                    # orig_mask_modal = orig_mask_modal * 255
                    # draw_img = sample.image
                    # color = (0,0,225)
                    # thickness = 2
                    # cv2.rectangle(draw_img, (a, b), (c, d), color, thickness)
                    #
                    # cv2.imwrite("show_grid.png", draw_img)
                    subimage_side_list.append(box_side)

        obj2subimage_size[object_lid] = subimage_side_list

    for obj_id in object_lids:
        temp = sum(obj2subimage_size[obj_id]) / len(obj2subimage_size[obj_id])
        factor = 420/temp
        print(obj_id,temp,factor)






def main() -> None:
    opts = config_util.load_opts_from_json_or_command_line(
        InferOpts
    )[0]
    infer(opts)


if __name__ == "__main__":
    main()