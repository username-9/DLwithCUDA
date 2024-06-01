import os

import cv2
import numpy as np
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn
from tqdm import tqdm


def to_single_band(img_file_path, out_file_dir):
    # 类别和对应的RGB值
    classes = {
        '不透水面': (255, 255, 255),
        '建筑物': (0, 0, 255),
        '低矮植被': (0, 255, 255),
        '树木': (0, 255, 0),
        '汽车': (255, 255, 0),
        '背景': (255, 0, 0)
    }

    # 逆映射RGB到类别的字典（整数标签）
    rgb_to_label = {v: k for k, v in enumerate(classes.values(), start=0)}  # 假设我们跳过0作为背景或未知类别

    # 读取原始PNG图像
    bgr_img = cv2.imread(img_file_path)
    img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    # 初始化一个与原始图像大小相同的单通道图像，用于存储类别标签，使用-1表示未知类别
    class_map = np.full(img.shape[:2], -1, dtype=np.uint8)

    # 遍历类别和RGB值
    for label, rgb in classes.items():
        # 使用NumPy的广播功能创建一个与原始图像形状相同的掩码
        mask = (img[:, :, 0] == rgb[0]) & (img[:, :, 1] == rgb[1]) & (img[:, :, 2] == rgb[2])
        # 将找到的像素设置为对应的类别标签
        class_map[mask] = rgb_to_label[rgb]

    # 如果你希望将0作为背景或未知类别的标签
    # class_map[class_map == -1] = 0

    if not os.path.exists(out_file_dir):
        os.makedirs(out_file_dir)
    out_put_path = os.path.join(out_file_dir, str(img_file_path.split('\\')[-1]))
    # 保存单通道分类图
    cv2.imwrite(out_put_path, class_map)

    # # 创建一个颜色映射（可选），用于可视化
    # color_labels = np.array(list(classes.values()))
    # color_map = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # for i, color in enumerate(color_labels, start=1):  # 从1开始，因为0可能用于背景或未知类别
    #     color_map[class_map == i] = color
    # if not os.path.exists(os.path.join(out_file_dir, "color_map")):
    #     os.mkdir(os.path.join(out_file_dir, "color_map"))
    # color_map_path = os.path.join(out_file_dir, "color_map", str(img_file_path.split('\\')[-1]))
    # # 保存伪彩色图像（可选）
    # cv2.imwrite(color_map_path, color_map)


if __name__ == '__main__':
    os.chdir(r"C:\Users\PZH\Desktop\RGZNYY\Data\test\label")
    file_list = os.listdir(".")
    for file in file_list:
        if not file.endswith(".png"):
            file_list.remove(file)
    for file in tqdm(file_list, total=len(file_list), colour="WHITE"):
        to_single_band(file, "..\\SingleBand")
