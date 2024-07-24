import os
import torch
import pickle

pkl_folder_path = '/home/exp2/Diffusion_Model/GDSS-master/data'

# 检索所有.pkl文件
pkl_files = [f for f in os.listdir(pkl_folder_path) if f.endswith('.pkl')]

# 循环加载每个.pkl文件并打印输出
for file in pkl_files:
    data_path = os.path.join(pkl_folder_path, file)
    with open(data_path, 'rb') as file:
        data = pickle.load(file)
        print(f"Loaded {file.name}:")
        print(data)
        print("--------------------------")

info_files = [f for f in os.listdir(info_folder_path) if f.endswith('.pkl')]
# 循环加载每个.pkl文件并打印输出
for file in info_files:
    data_path = os.path.join(info_folder_path, file)
    with open(data_path, 'rb') as file:
        data = pickle.load(file)
        print(f"Loaded {file.name}:")
        print(data)
        print("--------------------------")
