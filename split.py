#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2013-2021, China University of Petroleum
@Contact : zhoutao@s.upc.edu.cn
@Software: PyCharm
@File    : split
@Time    : 2021/09/11 11:39
@Desc    : 数据划分
"""
import os
import shutil
import random
import numpy as np


def read_image_name_from_dir(file_root_dir: str):
    """
    读取路径下的图片名称，以及数量
    :param file_root_dir: 图片路径
    :return:
    """
    if os.path.exists(file_root_dir):
        img_name_list = os.listdir(file_root_dir)
        return img_name_list, len(img_name_list)
    else:
        return None, 0
    pass


def shuffle(x: list):
    """
    随机打乱列表
    :param x: 需要打乱列表
    :return: 返回打乱的列表
    """
    if type(x) == list:
        random.shuffle(x)
    else:
        print("Input data type isn't list")
    return x
    pass


def is_exists_else_create(dest_path: str):
    """
    判断文件是否存在，不存在则创建该文件
    :param dest_path: 文件夹路径
    :return: dest_path 返回该路径
    """
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    return dest_path


def get_split_slices_from_list(image_num_list: list, total: int):
    """
    从数组中生成切分索引列表
    :param total: 总的数据量
    :param image_num_list: 切分节点图片数量数组
    :return:
    """
    tmp_slices: list[int] = [0]
    for index, image_num in enumerate(image_num_list):
        tmp_slices.append(
            image_num + tmp_slices[index]
        )
    if tmp_slices[-2] > total:
        print("Input list invalid, please must ensure sum(image_num_list) not greater than total")
        return None
    if tmp_slices[-1] > total:
        tmp_slices[-1] = total
    split_slices = [(start, end) for start, end in zip(tmp_slices[0:-1], tmp_slices[1:])]
    return split_slices


def get_split_slices_from_num(image_num: int, slice_num: int, total: int):
    """
    从整数中生成切分索引列表
    :param image_num: 切分节点图片数量
    :param slice_num: 切分个数
    :param total: 总的数据量
    :return:
    """
    image_num_list = [image_num for _ in range(slice_num)]
    return get_split_slices_from_list(image_num_list, total)


def split(config: dict):
    """
    划分数据集
    :param config: 划分数据集的配置
    :return:
    """
    data_dir_path = config.get("data_dir_path")
    out_dir_path = config.get("out_dir_path")
    repeated = config.get("repeated")
    classes = config.get("classes")
    # 遍历字典，获取每一类的名称及其每个节点数量，按照类别名称进行划分处理
    for class_name, slices in classes.items():
        # 获取文件名称列表，以及文件数量
        class_file_name_list, class_file_count = read_image_name_from_dir(
            os.path.join(data_dir_path, class_name)
        )
        # 按照列表进行文件划分
        if isinstance(slices, list):
            if repeated != len(slices):
                print(f"Error: Input classes num is list, but repeated != slices.length. !!!")
                return
            # 生成数据划分的切片
            split_slices_list = get_split_slices_from_list(slices, total=class_file_count)
        # 按照数字进行均匀的划分
        elif isinstance(slices, int):
            # 生成数据划分的切片
            split_slices_list = get_split_slices_from_num(slices, repeated, total=class_file_count)
        else:
            print("Input classes invalid, please input it with int or list type")
            return

        # 循环处理图片
        for index, (start, end) in enumerate(split_slices_list):
            # 划分数据的节点路径
            node_dir_path = is_exists_else_create(os.path.join(out_dir_path, f"{index + 1}", class_name))
            node_file_name_list = class_file_name_list[start:end]
            print(
                f"Key: {class_name:16}, Node： {(index + 1):4}, "
                f"Split: split data from {start:4} to {end:4}, Total: {end - start}"
            )
            for filename in node_file_name_list:
                src_path = os.path.join(data_dir_path, class_name, filename)
                dst_path = os.path.join(node_dir_path, filename)
                shutil.copy(src_path, dst_path)
            pass
        pass
    pass


def get_class_name(data_dir_path: str):
    """
    获取到指定的数据类别
    :param data_dir_path: 文件夹路径
    :return:
    """
    classes = os.listdir(data_dir_path)
    return classes


def iid(config: dict):
    config["classes"].clear()
    data_dir_path = config.get("data_dir_path")
    classes = get_class_name(data_dir_path)
    repeated = config.get("repeated")
    iid_split_matrix = np.zeros((len(classes), repeated), dtype=int)
    compensate_index = 0
    for class_idx, class_name in enumerate(classes):
        _, class_file_count = read_image_name_from_dir(
            os.path.join(data_dir_path, class_name)
        )
        class_split_list = []
        iid_class_count = class_file_count // repeated
        left_class_count = class_file_count % repeated
        # 生成每个节点的图片数量
        for index in range(repeated):
            class_split_list.append(iid_class_count)
        # 每个类别多余的数量按照顺序给每个节点加上
        for index in range(left_class_count):
            plus_index = (index + compensate_index) % repeated
            class_split_list[plus_index] += 1
        compensate_index += left_class_count
        iid_split_matrix[class_idx] = class_split_list
    for class_idx, class_name in enumerate(classes):
        config["classes"][class_name] = list(iid_split_matrix[class_idx])
    split(config)
    pass


def get_classes_num_dict(data_dir_path: str):
    """
    获取到指定路径下子文件中的文件数量
    :param data_dir_path: 文件路径
    :return:
    """
    classes_image_num_dict = {}
    classes = get_class_name(data_dir_path)
    total_classes_num = 0
    for class_name in classes:
        class_name_num = os.path.join(data_dir_path, class_name)
        total_classes_num += len(os.listdir(class_name_num))
        classes_image_num_dict[class_name] = len(os.listdir(class_name_num))
    return total_classes_num, classes_image_num_dict
    pass


def non_iid(config: dict):
    config["classes"].clear()
    data_dir_path = config.get("data_dir_path")
    classes = get_class_name(data_dir_path)
    repeated = config.get("repeated")
    # 获取到每个类别的样本数量字典以及总样本数量
    total_classes_num, classes_image_num_dict = get_classes_num_dict(data_dir_path)
    non_iid_count = total_classes_num // repeated
    left_non_iid_count = total_classes_num - (repeated - 1) * non_iid_count
    # 生成每个节点划分后的样本数量
    node_num_list = [non_iid_count] * (repeated - 1) + [left_non_iid_count]
    # 生成划分的二维矩阵，从左至右为节点，从上至下为类别
    non_iid_split_matrix = np.zeros((len(classes), repeated), dtype=int)

    # 总共有repeated个节点
    for index in range(repeated):
        class_name, class_num = min(classes_image_num_dict.items(), key=lambda x: x[1] if x[1] > 0 else float("inf"))
        # 如果该类别的数量不够，则需要其他类别进行补偿
        if class_num < node_num_list[index]:
            non_iid_split_matrix[classes.index(class_name)][index] = class_num
            classes_image_num_dict[class_name] = 0
            # 需要补偿的图片数量
            compensate_num = node_num_list[index] - class_num
            while compensate_num > 0:
                # 获取补偿的类别数量进行修改
                compensate_class_name, compensate_class_num = max(classes_image_num_dict.items(), key=lambda x: x[1])
                # 进行补偿类别的数量足够，则直接进行补偿操作
                if compensate_class_num > compensate_num:
                    non_iid_split_matrix[classes.index(compensate_class_name)][index] = compensate_num
                    classes_image_num_dict[compensate_class_name] -= compensate_num
                    compensate_num = 0
                else:
                    non_iid_split_matrix[classes.index(compensate_class_name)][index] = compensate_class_num
                    classes_image_num_dict[compensate_class_name] = 0
                    # 修改补齐后的数量，依旧不等于0
                    compensate_num -= compensate_class_num
        else:
            non_iid_split_matrix[classes.index(class_name)][index] = class_num
            classes_image_num_dict[class_name] = int(node_num_list[index] - class_num)
        pass
    for index, class_name in enumerate(classes):
        config["classes"][class_name] = list(non_iid_split_matrix[index])
    split(config)
    pass


if __name__ == "__main__":
    cfg = {
        "data_dir_path": r"E:\PyCharm\data_fl_split\datasets\mnist\train",
        "out_dir_path": r"E:\PyCharm\data_fl_split\datasets\mnist\iid",
        "repeated": 10,
        "classes": {
            "COVID": 1080,
            "Lung_Opacity": 1800,
            "Normal": [1045, 1045, 1045, 325, 325, 325, 1720, 1720, 1720],
            "Viral Pneumonia": 405
        }
    }
    iid(cfg)
    # non_iid(cfg)
    pass
