from torch.utils.data import TensorDataset, DataLoader
from data.data_generators import load_dataset
from utils.graph_utils import init_features, graphs_to_tensor
import networkx as nx
import torch
import random
import os


def graphs_to_dataloader(config, graph_list):

    adjs_tensor = graphs_to_tensor(graph_list, config.data.max_node_num)
    x_tensor = init_features(config.data.init, adjs_tensor, config.data.max_feat_num)

    train_ds = TensorDataset(x_tensor, adjs_tensor)
    train_dl = DataLoader(train_ds, batch_size=config.data.batch_size, shuffle=True)
    return train_dl


# def dataloader(config, get_graph_list=False):
#     graph_list = load_dataset(data_dir=config.data.dir, file_name=config.data.data)
#     test_size = int(config.data.test_split * len(graph_list))
#     train_graph_list, test_graph_list = graph_list[test_size:], graph_list[:test_size]
#     if get_graph_list:
#         return train_graph_list, test_graph_list
#
#     return graphs_to_dataloader(config, train_graph_list), graphs_to_dataloader(config, test_graph_list)

# def graphs_to_dataloader(config, graph_list):
#
#     adjs_tensor = graphs_to_tensor(graph_list, config.data.max_node_num)
#     x_tensor = init_features(config.data.init, adjs_tensor, config.data.max_feat_num)
#     labels = [graph.graph['label'] for graph in graph_list]
#     y_tensor = torch.tensor(labels, dtype=torch.long)
#     #print(y_tensor.min(), y_tensor.max(), 'y_tensor.min(), y_tensor.max()')
#     #assert False
#
#     train_ds = TensorDataset(x_tensor, adjs_tensor, y_tensor)
#     #for i, tensor in enumerate(train_ds.tensors):
#     #    print(f"Tensor {i} shape: {tensor.shape}")
#     #for i in range(len(train_ds)):
#     #    sample = train_ds[i]
#     #    print(f"Sample {i}: {sample}")
#     #    if i==2:
#     #        assert False
#     #assert False
#     train_dl = DataLoader(train_ds, batch_size=config.data.batch_size, shuffle=True)
#     #for batch_idx, batch in enumerate(train_dl):
#     #    print(f"Batch {batch_idx}:")
#     #    for i, tensor in enumerate(batch):
#     #        print(f"  Tensor {i} shape: {tensor.shape}")
#     #    assert False
#     return train_dl

def dataloader(config, get_graph_list=False):
    graph_list = load_dataset(data_dir=config.data.dir, file_name=config.data.data)
    train_file_path = os.path.join('/home/exp2/Diffusion_Model/GDSS-master/data/', str(config.data.data) + '_train_indices.txt')
    test_file_path = os.path.join('/home/exp2/Diffusion_Model/GDSS-master/data/', str(config.data.data) + '_test_indices.txt')
    # print(test_file_path)
    # val_file_path = os.path.join('data/', str(config.data.dataset) + '_val_indices.txt')
    if not os.path.exists(train_file_path) or not os.path.exists(test_file_path):
    # if not os.path.exists(train_file_path) or not os.path.exists(test_file_path) or not os.path.exists(val_file_path):
        test_size = int(config.data.test_split * len(graph_list))
        # assume test_split equals to val_split
        # val_size = int(config.data.test_split * len(graph_list))
        # Generate a list of indices corresponding to the graph_list
        indices = list(range(len(graph_list)))
        # Shuffle the indices to create random splits
        random.shuffle(indices)
        # Split the indices into test and the rest (train + validation)
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]
        # val_indices = train_indices[:val_size]
        with open(train_file_path, 'w') as f1:
            f1.write(','.join(map(str, train_indices)) + '\n')
        with open(test_file_path, 'w') as f2:
            f2.write(','.join(map(str, test_indices)) + '\n')
        # with open(val_file_path, 'w') as f3:
        #     f3.write(','.join(map(str, val_indices)) + '\n')
        print('Index files were missing. Created new files. Please check dataset index!')
        #assert False
    else:
        with open(train_file_path, 'r') as f1:
            train_indices = list(map(int, f1.read().strip().split(',')))
        with open(test_file_path, 'r') as f2:
            test_indices = list(map(int, f2.read().strip().split(',')))
        # with open(val_file_path, 'r') as f3:
        #     val_indices = list(map(int, f3.read().strip().split(',')))
        train_graph_list = [graph_list[i] for i in train_indices]
        test_graph_list = [graph_list[i] for i in test_indices]
        print("train_graph_list && test_graph_list are Ready!")
        # val_graph_list = [graph_list[i] for i in val_indices]
        # print(len(train_graph_list), len(test_graph_list), len(val_graph_list))
        # print('len(train_graph_list), len(test_graph_list), len(val_graph_list)')
    if get_graph_list:
        return train_graph_list, test_graph_list

    return graphs_to_dataloader(config, train_graph_list), graphs_to_dataloader(config, test_graph_list)
