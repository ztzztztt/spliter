#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2013-2021, China University of Petroleum
@Contact : zhoutao@s.upc.edu.cn
@Software: PyCharm
@File    : cifar10
@Time    : 2021/09/10 14:58
@Desc    : cifar10数据集解压转换成图片
"""
import os
import cv2
import torchvision
import numpy as np


classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def unpickle(file):
    """cifar10 官方给出的解压函数"""
    import pickle
    with open(file, 'rb') as fo:
        dicts = pickle.load(fo, encoding='bytes')
    return dicts


def is_exists_else_create(dest_path: str):
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    return dest_path


def extract_image_from_dict(data_dict, save_dir_path):
    for image_num in range(10000):
        # 图片处理
        img = np.reshape(data_dict[b'data'][image_num], (3, 32, 32))
        img = np.transpose(img, (1, 2, 0))
        # 通道顺序修改BGR为RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 图片路径处理
        class_idx = data_dict[b"labels"][image_num]
        class_num = classes[class_idx]
        class_path = is_exists_else_create(os.path.join(save_dir_path, class_num))
        img_name = os.path.join(class_path, data_dict[b"filenames"][image_num].decode())
        cv2.imwrite(img_name, img)


def covert_cifar10_to_image(root_file_path):
    train_data_path = is_exists_else_create(os.path.join(root_file_path, "train"))
    test_data_path = is_exists_else_create(os.path.join(root_file_path, "test"))
    data_file_path = os.path.join(root_file_path, "cifar-10-batches-py")
    for batch_index in range(1, 6):
        cifar10_train_batch_name = os.path.join(data_file_path, 'data_batch_' + str(batch_index))
        data_dict = unpickle(cifar10_train_batch_name)
        print(cifar10_train_batch_name + ' is Loading')
        extract_image_from_dict(data_dict, train_data_path)
        print(f"{cifar10_train_batch_name} is Loaded")

    cifar10_test_batch_name = os.path.join(data_file_path, "test_batch")
    print(f"{cifar10_test_batch_name} is Loading")
    test_dict = unpickle(cifar10_test_batch_name)
    extract_image_from_dict(test_dict, test_data_path)
    print(f"{cifar10_test_batch_name} is Loaded")
    print('Finish Extraction from File to Image')


if __name__ == '__main__':
    train_cifar10 = torchvision.datasets.CIFAR10(
        root='datasets/cifar10',
        train=True,
        download=True
    )
    test_cifar10 = torchvision.datasets.CIFAR10(
        root='datasets/cifar10',
        train=False,
        download=True
    )
    # 输出数据集的信息
    print(train_cifar10)
    print(test_cifar10)
    covert_cifar10_to_image(os.path.join("datasets", "cifar10"))

