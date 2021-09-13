#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2013-2021, China University of Petroleum
@Contact : zhoutao@s.upc.edu.cn
@Software: PyCharm
@File    : mnist
@Time    : 2021/09/10 15:10
@Desc    : mnist数据集解压转换成图片
"""
import os
import cv2
import warnings
import torchvision
from tqdm import tqdm
from torchvision.datasets import mnist


def is_exists_else_create(dest_path: str):
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    return dest_path


def extract_image_from_dict(train_data, train_label, save_dir_path):
    total_size = int(train_data.shape[0])
    with tqdm(total=total_size, desc="Converting", unit="it") as bar:
        for index, (img, label) in enumerate(zip(train_data, train_label)):
            parent_img_path = is_exists_else_create(os.path.join(save_dir_path, str(label.item())))
            pic_path = os.path.join(parent_img_path, f"{index}.png")
            cv2.imwrite(pic_path, img.numpy())
            bar.update(1)
    pass


def convert_mnist_to_image(root_data_path):
    # 训练数据存放路径
    train_data_dir_path = is_exists_else_create(os.path.join(os.path.join(root_data_path, "train")))
    test_data_dir_path = is_exists_else_create(os.path.join(os.path.join(root_data_path, "test")))

    # 训练集
    train_data_path = os.path.join(root_data_path, "MNIST", "raw", "train-images-idx3-ubyte")
    train_label_path = os.path.join(root_data_path, "MNIST", "raw", "train-labels-idx1-ubyte")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_data = mnist.read_image_file(train_data_path)
        train_label = mnist.read_label_file(train_label_path)
    print("Extract Train Data")
    extract_image_from_dict(train_data, train_label, train_data_dir_path)
    # 测试集
    test_data_path = os.path.join(root_data_path, "MNIST", "raw", "t10k-images-idx3-ubyte")
    test_label_path = os.path.join(root_data_path, "MNIST", "raw", "t10k-labels-idx1-ubyte")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_data = mnist.read_image_file(test_data_path)
        train_label = mnist.read_label_file(test_label_path)
    print("Extract Test Data")
    extract_image_from_dict(train_data, train_label, test_data_dir_path)
    pass


if __name__ == '__main__':
    train_dataset = torchvision.datasets.MNIST(
        root='datasets/mnist/',
        train=True,
        download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root='datasets/mnist/',
        train=False,
    )
    convert_mnist_to_image(os.path.join("datasets", "mnist"))
    pass
