import pickle
import networkx as nx

#with open('grid.pkl', 'rb') as f:
  #  graphs = pickle.load(f)

#with open('ogbg_moltoxcast.pkl', 'rb') as f:
  #  graphs = pickle.load(f)

with open('ogbg-molfreesolv.pkl', 'rb') as f:
    graphs = pickle.load(f)

max_nodes = 0
min_nodes = 0
max_feature_count = 0
if isinstance(graphs, list):
    print(f"加载 {len(graphs)} 个 graphs.")
    for i, graph in enumerate(graphs):
        if isinstance(graph, nx.Graph):
            print(f"\n图 {i}:")
            print(f"Number of nodes: {graph.number_of_nodes()}")
            if max_nodes<graph.number_of_nodes():
                  max_nodes=graph.number_of_nodes()
            min_nodes = graph.number_of_nodes()
            if min_nodes>graph.number_of_nodes():
                  min_nodes=graph.number_of_nodes()

            for node in graph.nodes():
                 # 获取节点的特征字典
                 node_features = graph.nodes[node]
                 # 计算当前节点的特征数  都是0？？
                 feature_count = len(node_features)
                 # 更新最大特征数
                 if feature_count > max_feature_count:
                        max_feature_count = feature_count
            print(f"The number of features per node: {max_feature_count}")
            print(f"Number of edges: {graph.number_of_edges()}")
            print(f"Nodes: {list(graph.nodes())[:100]}")  
            print(f"Edges: {list(graph.edges())[:100]}")  
        else:
            print(f"Object at index {i} is not a NetworkX graph.")
else:
    print("No list.")

print(f"加载 {len(graphs)} 个 graphs.")
print(f"The maximum number of nodes in any graph within the pickle file is: {max_nodes}")
print(f"The minimum number of nodes in any graph within the pickle file is: {min_nodes}")
print(f"The maximum number of features per node: {max_feature_count}")