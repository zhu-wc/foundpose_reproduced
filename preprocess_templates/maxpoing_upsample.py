import cv2

import numpy as np

import os


def max_pooling(image, kernel_size=3, stride=3):
    """3x3 最大池化"""

    h, w = image.shape[:2]

    new_h = h // stride

    new_w = w // stride

    # 如果是彩色图像（3通道），单独处理每个通道

    if len(image.shape) == 3:

        pooled = np.zeros((new_h, new_w, image.shape[2]), dtype=image.dtype)

        for i in range(new_h):

            for j in range(new_w):

                for c in range(image.shape[2]):
                    window = image[i * stride: i * stride + kernel_size,

                             j * stride: j * stride + kernel_size, c]

                    pooled[i, j, c] = np.max(window)

    else:

        # 灰度图像

        pooled = np.zeros((new_h, new_w), dtype=image.dtype)

        for i in range(new_h):

            for j in range(new_w):
                window = image[i * stride: i * stride + kernel_size,

                         j * stride: j * stride + kernel_size]

                pooled[i, j] = np.max(window)

    return pooled


def bilinear_upsample(image, original_shape):
    """双线性插值上采样到原始尺寸"""

    h, w = original_shape[:2]

    return cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)


def process_images_in_directory(directory,output_dir):
    """处理目录下的所有图像"""

    for filename in os.listdir(directory):

        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):

            # 1. 读取图像

            img_path = os.path.join(directory, filename)

            img = cv2.imread(img_path)


            # 2. 最大池化 (3x3, stride=3)

            pooled = max_pooling(img, kernel_size=3, stride=3)

            # 3. 双线性插值上采样回原始尺寸

            upsampled = bilinear_upsample(pooled, img.shape)


            # 保存结果（可选）

            os.makedirs(output_dir, exist_ok=True)

            output_path = os.path.join(output_dir, f"{filename}")

            cv2.imwrite(output_path, upsampled)




# 示例：处理当前目录下的图像

if __name__ == "__main__":
    obj_list = [1,5,6,8,9,10,11,12]
    for id in obj_list:

        input_dir = "/home/zhu/zwc/FoundPose/bop_datasets/templates/v1/lmo/"+str(id)+"/rgb/"  # 替换为你的图像目录路径
        output_dir = "/home/zhu/zwc/FoundPose/bop_datasets/templates/v1/lmo/"+str(id)+"/preprocessed/"
        process_images_in_directory(input_dir,output_dir)