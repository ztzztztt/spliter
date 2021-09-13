#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2013-2021, China University of Petroleum
@Contact : zhoutao@s.upc.edu.cn
@Software: PyCharm
@File    : cifar100
@Time    : 2021/09/10 15:10
@Desc    : cifar100数据集解压转换成图片
"""
import os
import cv2
import torchvision
import numpy as np
from tqdm import tqdm


def unpickle(file):
    """cifar100 官方给出的解压函数"""
    import pickle
    with open(file, 'rb') as fo:
        dicts = pickle.load(fo, encoding='bytes')
    return dicts


def extract_image_from_dict(data_dict, label_dict, save_dir_path, label_fine: bool = True):
    length = data_dict[b'data'].shape[0]
    fine_label = label_dict[b'fine_label_names']
    coarse_label = label_dict[b'coarse_label_names']
    with tqdm(total=length, desc="Converting", unit="it") as bar:
        for i in range(0, length):
            # data_dict['data']为图片二进制数据, 读取image
            img = np.reshape(data_dict[b'data'][i], (3, 32, 32))
            img = np.transpose(img, (1, 2, 0))
            if label_fine:
                label_image = fine_label[data_dict[b'fine_labels'][i]].decode()
            else:
                label_image = coarse_label[data_dict[b'coarse_labels'][i]].decode()
            # 生成图片的保存路径
            is_exists_else_create(os.path.join(save_dir_path, label_image))
            pic_name = os.path.join(
                save_dir_path,
                label_image,
                f"{data_dict[b'filenames'][i].decode()}"
            )
            cv2.imwrite(pic_name, img)
            bar.update(1)


def is_exists_else_create(dest_path: str):
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    return dest_path


def convert_cifar100_to_image(root_file_path):
    # 原始数据目录
    target_dir = os.path.join(root_file_path, "cifar-100-python")
    # 图片保存目录
    train_root_dir = is_exists_else_create(os.path.join(root_file_path, "train"))
    test_root_dir = is_exists_else_create(os.path.join(root_file_path, "test"))
    # 获取label对应的class，分为20个coarse class，共100个 fine class
    meta_dict = unpickle(os.path.join(target_dir, "meta"))
    train_data = unpickle(os.path.join(target_dir, "train"))
    # 生成训练集图片
    print("Train Dataset is Loading...")
    extract_image_from_dict(train_data, meta_dict, train_root_dir)
    print("Train Dataset Loaded.")
    # 生成测试集图片
    print("Test Dataset is Loading...")
    test_data = unpickle(os.path.join(target_dir, "test"))
    extract_image_from_dict(test_data, meta_dict, test_root_dir)
    print("Test DataSet Loaded")
    pass


if __name__ == '__main__':
    train_cifar100 = torchvision.datasets.CIFAR100(
        root='datasets/cifar100',
        train=True,
        download=True
    )
    test_cifar100 = torchvision.datasets.CIFAR100(
        root='datasets/cifar100',
        train=False,
        download=True
    )
    # 输出数据集的信息
    print(train_cifar100)
    print(test_cifar100)
    # 将图像从数据集中结构出来
    convert_cifar100_to_image(os.path.join("datasets", "cifar100"))
