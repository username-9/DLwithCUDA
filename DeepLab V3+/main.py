# main program
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import data_deal
import deepLab_v3plus
from demo import min_max_to_one


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist


def scores(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0  # added
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Frequency Weighted IoU": fwavacc,
        "Mean IoU": mean_iu,
        "Class IoU": cls_iu,
    }


def test(epoch_id, model_name, test_data_path: str, device: torch.device):
    """
    Evaluation on validation set
    """
    predictions, gts = [], []
    image_ids = 0
    test_data = data_deal.UsingOwnData(test_data_path)
    test_data_loader = torch.utils.data.DataLoader(test_data)
    pro_bar = tqdm(test_data_loader, total=test_data.__len__(), position=0, leave=True)
    pro_bar.set_description(f"Validation after epoch {epoch_id+1}: ")
    for image, gt_labels in pro_bar:
        # Image
        image_ids += 1
        image = image.to(device)

        # Forward propagation
        log_items = model_name(image)

        # Save on disk for CRF post-processing
        # for logit in log_items:
        #    filename = os.path.join("log_items" + str(image_ids) + ".npy")
        #    np.save(filename, logit.detach().numpy())

        # Pixel-wise labeling
        # _, h, w = gt_labels.shape
        # log_items = F.interpolate(
        #     log_items, size=(H, W), mode="bilinear", align_corners=False
        # )
        probs = F.softmax(log_items, dim=1)
        labels = torch.argmax(probs, dim=1)
        array_1 = labels.cpu().numpy()
        # b, W, H = np.shape(array_1)
        # for w in range(W):
        #     for h in range(H):
        #         if 0 <= array_1[0, w, h] <= 0.5:
        #             array_1[0, w, h] = 0
        #         else:
        #             array_1[0, w, h] = 1
        predictions += list(array_1)
        # gt_labels = torch.argmax(gt_labels, dim=0)
        gts += list(gt_labels.numpy())

    # Pixel Accuracy, Mean Accuracy, Class IoU, Mean IoU, Freq Weighted IoU
    score = scores(gts, predictions, n_class=6)
    del predictions, gts

    save_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "scores"
    )
    save_path = os.path.join(save_dir, f"{str(epoch_id + 1)} scores.json")

    with open(save_path, "w") as f:
        json.dump(score, f, indent=4, sort_keys=True)


def train(epoch_all: int, model_name, train_data_path: str = r"C:\Users\PZH\Desktop\RGZNYY\Data",
          val_data_path: str = r"C:\Users\PZH\Desktop\RGZNYY\Data\test"):
    if torch.cuda.is_available():
        print("CUDA is available! Training on GPU ...")
        device = torch.device("cuda")  # 设备对象，表示GPU
    else:
        print("CUDA is not available! Training on CPU ...")
        device = torch.device("cpu")  # 设备对象，表示CPU

    model_name.to(device)
    loss = torch.nn.CrossEntropyLoss()
    loss.to(device)
    learn_step = 0.001
    optimize = torch.optim.SGD(model_name.parameters(), lr=learn_step)
    epoch_num = epoch_all
    model_name.train()
    train_data_0 = data_deal.UsingOwnData(train_data_path)
    # train_data_0 = data_deal.UsingOwnData(r"C:\Users\PZH\Desktop\RGZNYY\Data")
    test_data_path = val_data_path
    # test_data = data_deal.UsingOwnData(r"C:\Users\PZH\Desktop\RGZNYY\Data\test")
    # my_data = data_deal.UsingOwnData(r"C:\Users\PZH\Desktop\RGZNYY\Data\pre")
    # test_data = data_deal.UsingOwnData(r"C:\Users\PZH\Desktop\RGZNYY\Data\pre\test")
    train_data = DataLoader(dataset=train_data_0, batch_size=1, shuffle=True)
    # test_data_loader = DataLoader(dataset=test_data)
    train_step = 0
    # 训练
    for i in range(epoch_num):
        re_loss_ls = []
        total = int(train_data_0.__len__())
        process_bar = tqdm(train_data, total=total, position=0, leave=True)
        process_bar.set_description(f"Epoch {i + 1} of {epoch_all}")
        for data in process_bar:
            # start_time = time.time()
            images, targets = data
            images = images.to(device)
            targets = targets.to(device)
            outputs = model_name(images)
            result_loss = loss(outputs, targets)
            optimize.zero_grad()
            result_loss.backward()
            optimize.step()
            train_step += 1
            # end_time = time.time()
            if train_step % 100 == 0:
                process_bar.set_postfix(dict(loss=f"{result_loss.item():.5f} at {train_step}"))
                re_loss_ls.append(result_loss.item())
        for num in re_loss_ls:
            print(num, end=' ')
            if (re_loss_ls.index(num)+1) % 5 == 0:
                print("\n")
        print(f"Min Loss: {min(re_loss_ls)}")
        # if (i + 1) % 500 == 0:
        if (i + 1) % 1 == 0:
            # 保存模型
            torch.save(model_name.state_dict(), "my_model_{}.pth".format((i + 1)))
        model.eval()  # 在验证状态
        test(i, model, val_data_path, device)


def train_go_on(my_model: torch.nn.Module, model_parameters, epoch_go_on: int,
                train_data_path: str, val_data_path: str):
    """
        Load model parameters and train continue
        :param model_parameters: model parameters' path
        :param my_model: model to train
        :param epoch_go_on: epoch number
        :param train_data_path: training data path
        :param val_data_path: validation data path
    """
    model_path = model_parameters
    my_model.load_state_dict(torch.load(model_path))
    train(epoch_go_on, my_model, train_data_path, val_data_path)


if __name__ == '__main__':
    model = deepLab_v3plus.DeepLabV3Plus(n_classes=6, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18],
                                         multi_grids=[1, 2, 4], output_stride=16)
    # train(10, model, r"C:\Users\PZH\Desktop\RGZNYY\Data",
    #       r"C:\Users\PZH\Desktop\RGZNYY\Data\test")
    train_go_on(model, r"C:\Users\Administrator\PycharmProjects\DLwithCUDA\DeepLab V3+\my_model_8.pth",
                10,  r"C:\Users\PZH\Desktop\RGZNYY\Data",
                r"C:\Users\PZH\Desktop\RGZNYY\Data\test")
