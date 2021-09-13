#!/usr/bin/env python 
# -*- encoding: utf-8 -*- 
"""
@Author  : zhoutao
@License : (C) Copyright 2013-2017, China University of Petroleum
@Contact : zhoutao@s.upc.edu.cn
@Software: PyCharm
@File    : data_split.py 
@Time    : 2021/4/22 下午9:42 
@Desc    : 
"""
import array
import os
import shutil
import random


def read_img_name_from_dir(file_root_dir: str):
    if os.path.exists(file_root_dir):
        img_name_list = os.listdir(file_root_dir)
        return img_name_list, len(img_name_list)
    else:
        return None, 0
    pass


def shuffle_array(array: list):
    if type(array) == list:
        random.shuffle(array)
    return array
    pass


def split(config: dict):
    data_root = config.get("data_root")
    out_root = config.get("out_root")
    repeated = config.get("repeated")
    classes = config.get("classes")
    # 遍历字典，获取每一类的名称及其每个节点数量
    for key, num_list_or_int in classes.items():
        # 获取文件名称列表，以及文件数量
        class_file_name_list, class_file_count = read_img_name_from_dir(
            os.path.join(data_root, key)
        )
        # 生成文件下标
        class_file_index_array = list(range(0, class_file_count))
        # 随机打乱文件下标
        shuffle_array(class_file_index_array)
        # 如果按照列表进行文件划分
        if isinstance(num_list_or_int, list):
            if repeated != 1:
                print(f"Error: input classes num is list, but repeated != 1. !!!")
                return
            start = 0
            end = 0
            for index, value in enumerate(num_list_or_int):
                # 生成当前节点切分开始与结束的下标
                start = end
                end = start + value
                node_path = os.path.join(out_root, f"{index + 1}", key)
                if not os.path.exists(node_path):
                    os.makedirs(node_path)
                node_file_name_list = class_file_name_list[start:end]
                print(f"Node： {(index + 1):4}， Key: {key}, "
                      f"Split: index from {start:4} to {end:4}, total {end - start}")
                for filename in node_file_name_list:
                    src_path = os.path.join(data_root, key, filename)
                    dst_path = os.path.join(node_path, filename)
                    shutil.copy(src_path, dst_path)
            # 剩下的文件全部为测试集
            if start <= end <= class_file_count:
                start = end
                end = class_file_count
                node_path = os.path.join(out_root, "test", key)
                if not os.path.exists(node_path):
                    os.makedirs(node_path)
                node_file_name_list = class_file_name_list[start:end]
                print(f"Node： test， Key: {key}, "
                      f"Split: index from {start:4} to {end:4}, total {end - start}")
                for filename in node_file_name_list:
                    src_path = os.path.join(data_root, key, filename)
                    dst_path = os.path.join(node_path, filename)
                    shutil.copy(src_path, dst_path)
            print("====================================================================")
        # 按照数字进行均匀的划分
        elif isinstance(num_list_or_int, int):
            # 遍历节点进行数据划分
            for node_i in range(repeated + 1):
                start = node_i * num_list_or_int
                if node_i == repeated:
                    end = class_file_count
                    node_path = os.path.join(out_root, "test", key)
                else:
                    end = node_i * num_list_or_int + num_list_or_int
                    node_path = os.path.join(out_root, f"{node_i + 1}", key)
                if not os.path.exists(node_path):
                    os.makedirs(node_path)
                node_file_name_list = class_file_name_list[start:end]

                print(f"Key: {key}, Split: index from {start:4} to {end:4}, total {end - start}")
                for filename in node_file_name_list:
                    src_path = os.path.join(data_root, key, filename)
                    dst_path = os.path.join(node_path, filename)
                    shutil.copy(src_path, dst_path)
            print("====================================================================")
        else:
            print("input classes value invalid, please input it with int or list type")
    pass


if __name__ == "__main__":
    cfg = {
        "data_root": "/home/chase/zhoutao/data/noniid/data",
        "out_root": "/home/chase/zhoutao/data/noniid",
        "repeated": 1,
        "classes": {
            "COVID": 1080,
            "Lung_Opacity": 1800,
            "Normal": [1045, 1045, 1045, 325, 325, 325, 1720, 1720, 1720],
            "Viral Pneumonia": 405
        }
    }
    split(cfg)
    pass
