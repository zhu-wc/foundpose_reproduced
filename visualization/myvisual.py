import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh


from utils import (
    projector_util,
    repre_util,
    vis_base_util,
    renderer_base,
    render_vis_util,
    logging,
    structs,
    misc,
    geometry
)

from utils.misc import tensor_to_array, array_to_tensor

def add_subimage_temps(
    mask_template_list: list,
    base_image: np.ndarray,
    object_repre: repre_util.FeatureBasedObjectRepre,
    object_lid: int,
    object_pose_m2w: structs.ObjectPose,
    object_pose_m2w_gt: Optional[structs.ObjectPose],
    feature_map_chw: torch.Tensor,
    feature_map_chw_proj: torch.Tensor,
    vis_feat_map: bool,
    object_box: np.ndarray,
    object_mask: np.ndarray,
    camera_c2w: structs.CameraModel,
    corresp: Dict,
    matched_template_ids: List[int],
    matched_template_scores: List[float],
    best_template_ind: int,
    renderer: renderer_base.RendererBase,
    pose_eval_dict: Dict,
    corresp_top_n: int = 50,
    dpi: int = 100,
    inlier_thresh: float = 10,
    object_pose_m2w_coarse: Optional[structs.ObjectPose] = None,
    pose_eval_dict_coarse: Optional[Dict] = None,
    vis_for_paper: bool = True,
    vis_for_teaser: bool = False,
    extractor: Any = None,
):

    device = feature_map_chw.device

    accent_color = (255, 255, 255)
    # accent_color = (0, 255, 0)
    if vis_for_paper:
        vis_margin = 0
        match_alpha = 0.8
    elif vis_for_teaser:
        vis_margin = 8
        match_alpha = 0.8
    else:
        vis_margin = 0
        match_alpha = 0.5

    # For visualization for paper.
    paper_gray = 230
    paper_bg_thresh = 5

    image_height = base_image.shape[0]
    image_width = base_image.shape[1]

    # if vis_feat_map:
    #     base_image_vis = vis_pca_feature_map(
    #         feature_map_chw=feature_map_chw_proj,
    #         image_height=image_height,
    #         image_width=image_width,
    #         pca_projector=object_repre.projector_vis,
    #     )
    # else:
    base_image_vis = (0.4 * base_image).astype(np.uint8)

    vis_tiles = []

    template_id = int(matched_template_ids[best_template_ind])  # 获取生成最佳姿态估计结果时所使用的模板id
    template = object_repre.templates[template_id] #利用模板id取出模板，模板的shape=(3,420,420)

    query_feat_vis = None
    template_feat_vis = None


    # ------------------------------------------------------------------------------
    # Row 1: Query image with object poses
    # ------------------------------------------------------------------------------

    if not vis_for_teaser:

        # Row 1 left:

        # Show object mask. 根据CNOS实例预测裁减后的测试图像，并在上面叠加CNOS预测到的实例掩码
        mask_3c = np.tile(np.expand_dims(object_mask.astype(np.float32), -1), (1, 1, 3))
        mask_3c_bool = mask_3c.astype(bool)
        if vis_for_paper:
            vis = np.array(base_image).astype(np.float32)
            vis[mask_3c_bool] *= 0.5 # 在子图中与实例掩码对应部分的像素处，添加指定颜色*0.5
            vis[mask_3c_bool] += np.tile(
                0.5 * np.array(accent_color), (int(np.sum(object_mask)), 1)
            ).flatten() #
            vis = vis.astype(np.uint8)
        else:
            vis = np.array(base_image_vis)
            vis[mask_3c_bool] = 0.5 * vis[mask_3c_bool] + 0.5 * 255
            vis = vis.astype(np.uint8)

        vis_base_util.plot_images(imgs=[vis], dpi=dpi)
        if not vis_for_paper:
            vis_base_util.add_text(0, "Input mask")
        # vis_base_util.plot_boundingbox(object_box)
        vis_tile_left = vis_base_util.save_plot_to_ndarray()

        # ROW 1 right
        matched_templates = [
            np.asarray(object_repre.templates[i].astype(np.float32)/255.0) # 归一化到区间[0,1]内
            for i in matched_template_ids
        ]
        matched_templates = [np.transpose(t, (1, 2, 0)) for t in matched_templates] # 调整维度的顺序，改为(420,420,3)

        # 加载模板对应的掩码，将其中元素置为0和1，转化为布尔类型。子图中对应位置处的像素值*0.5,模板图对应位置处的像素值*0.5,求和。
        add_list = []
        for i in range(5):
            template_mask = (mask_template_list[i]/255).astype(np.uint8)

            contours, _ = cv2.findContours(template_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            height, width = template_mask.shape
            contour_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(contour_mask, contours, -1, 50, thickness=5)
            contour_color = np.tile(np.expand_dims(contour_mask.astype(np.float32), -1), (1, 1, 3))
            contour_mask = (contour_color/255).astype(bool)

            vis_i = np.array(base_image).astype(np.float32)

            vis_i[contour_mask] =  contour_color[contour_mask]
            vis_i = vis_i.astype(np.uint8)
            #
            vis_base_util.plot_images(imgs=[vis_i], dpi=dpi)
            result_i = vis_base_util.save_plot_to_ndarray()
            add_list.append(result_i)

        # 左+右 -> 合成完整的一行图像
        tile = np.hstack([vis_tile_left, add_list[0]])
        vis_tiles.append(tile)
        tile = np.hstack([add_list[1], add_list[2]])
        vis_tiles.append(tile)
        tile = np.hstack([add_list[3], add_list[4]])
        vis_tiles.append(tile)
        return vis_tiles


