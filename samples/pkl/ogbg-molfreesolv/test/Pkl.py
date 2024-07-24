import pickle
import networkx as nx

#with open('grid.pkl', 'rb') as f:
  #  graphs = pickle.load(f)

with open('Jun11-14:03:38-sample.pkl', 'rb') as f:
    graphs = pickle.load(f)

if isinstance(graphs, list):
    print(f"加载 {len(graphs)} 个 graphs.")
    for i, graph in enumerate(graphs):
        if isinstance(graph, nx.Graph):
            print(f"\n图 {i}:")
            print(f"Number of nodes: {graph.number_of_nodes()}")
            print(f"Number of edges: {graph.number_of_edges()}")
            print(f"Nodes: {list(graph.nodes())[:100]}")  
            print(f"Edges: {list(graph.edges())[:100]}")  
        else:
            print(f"Object at index {i} is not a NetworkX graph.")
else:
    print("No list.")