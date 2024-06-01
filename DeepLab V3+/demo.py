import numpy as np
import torch
import cv2
from PIL.Image import Image
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

import data_deal
import deepLab_v3plus


def min_max_to_one(array):
    for i in range(np.shape(array)[0]):
        min_val = np.min(array[i])
        max_val = np.max(array[i])
        array[i] = (array[i] - min_val) / (max_val - min_val)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    '''
        加载模型与参数
    '''

    # 加载模型
    my_model = deepLab_v3plus.DeepLabV3Plus(n_classes=6, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18],
                                            multi_grids=[1, 2, 4], output_stride=16).to(device)
    model_path = r"my_model_5.pth"
    if device == "cpu":
        # 加载模型参数，权重文件传过来
        my_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        my_model.load_state_dict(torch.load(model_path))
    # 加载图片
    work_path = r"C:\Users\PZH\Desktop\RGZNYY\Data\pre\test\work"
    work_data = data_deal.WorkData(work_path)
    work_loader = DataLoader(dataset=work_data, shuffle=None)
    my_model.eval()
    with torch.no_grad():  # 验证的部分，不是训练所以不要带入梯度
        image_id = 0
        for images in tqdm(work_loader, total=work_loader.__len__()):
            image_id += 1
            outputs_ = my_model(images)
            # un_loader = transforms.ToPILImage()
            # image = torch.clone(outputs_)  # clone the tensor
            image = torch.argmax(outputs_, dim=1).to(device)
            image = image.numpy()
            image = np.squeeze(image,0)  # remove the fake batch dimension
            # 最大最小归一化
            # min_max_to_one(image)
            # image = np.transpose(image, [1, 2, 0])
            # H, W, C = np.shape(image)
            cv2.imwrite(fr"{work_path}\re\example{image_id}.png", image)
