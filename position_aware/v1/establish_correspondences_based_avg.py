from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision
from utils import (
    knn_util,
    repre_util,
    template_util,
    logging, misc
)

def cyclic_buddies_matching(
    query_points: torch.Tensor,
    query_features: torch.Tensor,
    query_knn_index: knn_util.KNN,
    object_features: torch.Tensor,
    object_knn_index: knn_util.KNN,
    top_k: int,
    debug: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Find best buddies via cyclic distance (https://arxiv.org/pdf/2204.03635.pdf)."""

    # Find nearest neighbours in both directions.
    query2obj_nn_ids = object_knn_index.search(query_features)[1].flatten() # query->template, 142个 匹配到的index
    obj2query_nn_ids = query_knn_index.search(object_features)[1].flatten() # template->query, 157个 匹配到的index

    # 2D locations of the query points.
    u1 = query_points #query中point的坐标,(142,2)

    # 2D locations of the cyclic points.
    cycle_ids = obj2query_nn_ids[query2obj_nn_ids] # query->template->query142个 匹配到的index
    u2 = query_points[cycle_ids] # query中point的重投影坐标

    # L2 distances between the query and cyclic points.
    cycle_dists = torch.linalg.norm(u1 - u2, axis=1)  #计算坐标的重投影误差

    # Keep only top k best buddies.
    top_k = min(top_k, query_points.shape[0])
    _, query_bb_ids = torch.topk(-cycle_dists, k=top_k, sorted=True) # 重投影误差最小的k个。由于k=142,所以该过程的主要功能是依照置信度排序

    # Best buddy scores.
    bb_dists = cycle_dists[query_bb_ids]
    bb_scores = torch.as_tensor(1.0 - (bb_dists / bb_dists.max())) # 计算重投影误差的归一化分数

    # Returns IDs of the best buddies.
    object_bb_ids = query2obj_nn_ids[query_bb_ids] # 依照重投影误差，对query->template的检索结果进行排序

    '''
    query中第query_bb_ids[0]个点与template中编号为object_bb_ids[0]的点匹配，二者的重投影距离为bb_dists[0]，距离得分为bb_scores[0]
    '''
    return query_bb_ids, object_bb_ids, bb_dists, bb_scores # 重投影距离，重投影误差分数

def find_nearest_object_features(
    query_features: torch.Tensor,
    knn_index: knn_util.KNN,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Find the nearest reference feature for each query feature.
    nn_dists, nn_ids = knn_index.search(query_features)

    knn_k = nn_dists.shape[1]

    # Keep only the required k nearest neighbors.
    nn_dists = nn_dists[:, :knn_k]
    nn_ids = nn_ids[:, :knn_k]

    # The distances returned by faiss are squared.
    nn_dists = torch.sqrt(nn_dists)

    return nn_ids, nn_dists

def calc_tfidf(
    feature_word_ids: torch.Tensor,
    feature_word_dists: torch.Tensor,
    word_idfs: torch.Tensor,
    soft_assignment: bool = True,
    soft_sigma_squared: float = 100.0,
) -> torch.Tensor:
    """Ref: https://www.di.ens.fr/~josef/publications/torii13.pdf"""

    device = feature_word_ids.device

    # Calculate soft-assignment weights, as in:
    # "Lost in Quantization: Improving Particular Object Retrieval in Large Scale Image Databases"
    if soft_assignment:
        word_weights = torch.exp(
            -torch.square(feature_word_dists) / (2.0 * soft_sigma_squared)
        )
    else:
        word_weights = torch.ones_like(feature_word_dists)

    # Normalize the weights such as they sum up to 1 for each query.
    word_weights = torch.nn.functional.normalize(word_weights, p=2, dim=1).reshape(-1) # 1/sqrt(3)=0.577

    # Calculate term frequencies.
    # tf = word_weights  # https://www.cs.cmu.edu/~16385/s17/Slides/8.2_Bag_of_Visual_Words.pdf
    tf = word_weights / feature_word_ids.shape[0]  # From "Lost in Quantization".

    # Calculate inverse document frequencies.
    feature_word_ids_flat = feature_word_ids.reshape(-1) # (588,)
    idf = word_idfs[feature_word_ids_flat]

    # Calculate tfidf values.
    tfidf = torch.multiply(tf, idf)

    # Construct the tfidf descriptor.
    num_words = word_idfs.shape[0]
    tfidf_desc = torch.zeros(
        num_words, dtype=word_weights.dtype, device=device
    ).scatter_add_(dim=0, index=feature_word_ids_flat.to(torch.int64), src=tfidf)

    return tfidf_desc


def my_establish_correspondences(
    # query_feature_cls,
    # templates_feature_matrix,
    # template_vector_metrix,
    templates_query_points,
    template_info,
    query_points: torch.Tensor,
    query_features: torch.Tensor,
    object_repre: repre_util.FeatureBasedObjectRepre,
    template_matching_type: str,
    feat_matching_type: str,
    top_n_templates: int,
    top_k_buddies: int,
    visual_words_knn_index: Optional[knn_util.KNN] = None,
    template_knn_indices: Optional[List[knn_util.KNN]] = None,
    debug: bool = False,
) -> List[Dict]:
    """Establishes 2D-3D correspondences by matching image and object features."""

    timer = misc.Timer(enabled=debug)
    timer.start()
    # 匹配5个最的相关的模板，返回值为模板的索引与置信度
    word_ids, word_dists = find_nearest_object_features( # 为实例中子图中的每一个有效描述符检索三个最邻近词向量
        query_features=query_features,
        knn_index=visual_words_knn_index,
    )

    all_word_index = word_ids.reshape(-1)
    unique_words_index, unique_count = torch.unique(all_word_index, return_counts=True)

    # 以均值的方式，计算当前图像每一个邻近单词的2维坐标
    frequence_map = unique_words_index.unsqueeze(-1) == all_word_index.unsqueeze(0)
    frequence_map = frequence_map.int().float()
    query_points_pos = query_points.unsqueeze(1).expand(-1, 3, -1).reshape(-1, 2)
    unique_words_pos = (frequence_map @ query_points_pos) / unique_count.unsqueeze(1)

    # 构建形状为(2048,2)的矩阵，储存当前图像中每一个单词对应的坐标。未出现单词的坐标为(0,0)
    all_words_pos_x = torch.zeros(2048).to(query_points.device).scatter_add_(dim=0, index=unique_words_index,src=unique_words_pos[:,0]).unsqueeze(-1)
    all_words_pos_y = torch.zeros(2048).to(query_points.device).scatter_add_(dim=0, index=unique_words_index,src=unique_words_pos[:,1]).unsqueeze(-1)
    all_words_pos = torch.cat([all_words_pos_x, all_words_pos_y], dim=1)

    #  构建形状为(2048,)的向量，储存当前图像中每一个单词出现的次数
    all_words_count = torch.zeros(2048).to(query_points.device).scatter_add_(dim=0, index=unique_words_index, src=unique_count.float())


    # 计算query与模板之间的距离
    dis_list = []
    for i in range(798):
        # 计算两幅图像中特征词索引的交集
        mask = torch.isin(unique_words_index,template_info[i]["index"])
        pair_words_index = unique_words_index[mask]

        # 当不存在特征词交集时，直接跳过。
        if(len(pair_words_index)==0):
            dis_list.append(torch.tensor(100000.0).to(query_points.device).unsqueeze(0))
            continue

        # 获取模板和子图中特征词的2维坐标
        c1 = template_info[i]["words_pos"][pair_words_index] # templates
        c2 = all_words_pos[pair_words_index] # subimage

        # 求和交集特征词在子图和模板中的频率，然后获取最大频率单词所在的位置
        fre2 = all_words_count[pair_words_index]
        fre2 = template_info[i]["count"][pair_words_index] + fre2
        _, max_fre_index = torch.max(fre2, dim=0)

        # 以max_fre_index处单词为原点，在两副图像上构建新的坐标系，并计算新坐标系下每一个局部视觉词的坐标
        new_c1 = c1 - c1[max_fre_index]
        new_c2 = c2 - c2[max_fre_index]

        # 在新坐标系下，计算每一对坐标的L2范数
        dis_L2 = torch.norm(new_c1 - new_c2, dim=1)

        # 获取单词出现的次数，单词次数向量与单词距离向量逐位相乘再求和，获取所有配对单词的距离之和。=》作为配对距离
        used_point_vector = (all_words_count[pair_words_index] + template_info[i]["count"][pair_words_index])
        paired_dis = torch.sum(used_point_vector*dis_L2)

        # 统计未配对patch的数量，然后与最大配对距离相乘 =》 作为惩罚距离
        used_point_count = torch.sum(used_point_vector)
        unused_point_count = query_features.shape[0] + templates_query_points[i].shape[0] - used_point_count
        unpaired_dis = unused_point_count * max(dis_L2)

        # 整合所有距离，然后除以两张图像中patch的数量 =》 距离平均值
        all_dis = paired_dis + unpaired_dis
        all_points = query_features.shape[0] + templates_query_points[i].shape[0]

        # 将子图与当前模板计算出的平均距离存入列表
        dis_list.append((all_dis/all_points).unsqueeze(0))

    # 距离越小认为匹配的程度越好，因此在上述计算结果前添加负号以调用topk方法
    temp = torch.cat(dis_list)
    temp = -temp
    template_scores, template_ids = torch.topk(temp,k=5,dim=0)

    timer.elapsed("Time for template matching")
    timer.start()
    # Build knn index for query features.
    query_knn_index = None
    if feat_matching_type == "cyclic_buddies":
        query_knn_index = knn_util.KNN(k=1, metric="l2")
        query_knn_index.fit(query_features)

    # Establish correspondences for each dominant template separately.
    corresps = []
    for template_counter, template_id in enumerate(template_ids):

        # Get IDs of features originating from the current template.
        tpl_feat_mask = torch.as_tensor(
            object_repre.feat_to_template_ids == template_id
        )
        tpl_feat_ids = torch.nonzero(tpl_feat_mask).flatten() #  在全部模板的特征描述符列表中，指定模板对应描述符的索引构成子列表

        # Find N best buddies.
        if feat_matching_type == "cyclic_buddies":
            assert object_repre.feat_vectors is not None
            '''
            query中第match_query_ids[0]个点与template中编号为match_obj_ids[0]的点匹配，
            二者的重投影距离为match_dists[0]，距离得分为match_scores[0]
            '''
            (
                match_query_ids, #query图像中point的索引,
                match_obj_ids, #template图像中point的索引,
                match_dists,
                match_scores,
            ) = cyclic_buddies_matching(
                query_points=query_points, # (142,2)
                query_features=query_features, # (142,256)
                query_knn_index=query_knn_index, # 存入了142个query描述符的实例,top1
                object_features=object_repre.feat_vectors[tpl_feat_ids], #指定模板的157个描述符，维度为256
                object_knn_index=template_knn_indices[template_id],
                top_k=top_k_buddies, # 300
                debug=debug,
            )
        else:
            raise ValueError(f"Unknown feature matching type ({feat_matching_type}).")

        match_obj_feat_ids = tpl_feat_ids[match_obj_ids] # 将单个template中point的索引，转化为在全部描述符集合中的索引

        # Structures for storing 2D-3D correspondences and related info.
        coord_2d = query_points[match_query_ids] # 142个query_point的2d坐标
        coord_2d_ids = match_query_ids # 142个query_point的索引
        assert object_repre.vertices is not None
        coord_3d = object_repre.vertices[match_obj_feat_ids] # template中point的3d坐标
        coord_conf = match_scores # 置信度。
        full_query_nn_dists = match_dists #重投影距离
        full_query_nn_ids = match_obj_feat_ids#当前template中point在全部template描述符集合中的索引
        nn_vertex_ids = match_obj_feat_ids#当前template中point在全部template描述符集合中的索引

        template_corresps = {
            "template_id": template_id,
            "template_score": template_scores[template_counter],
            "coord_2d": coord_2d,
            "coord_2d_ids": coord_2d_ids,
            "coord_3d": coord_3d,
            "coord_conf": coord_conf,
            "nn_vertex_ids": nn_vertex_ids,
        }
        # Add items for visualization/debugging.
        if debug:
            template_corresps.update(
                {
                    "nn_dists": full_query_nn_dists,
                    "nn_indices": full_query_nn_ids,
                }
            )

        corresps.append(template_corresps)

    timer.elapsed("Time for establishing corresp")

    return corresps
