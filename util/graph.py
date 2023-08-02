import networkx as nx


class Graph:
    def __init__(self, nodes, edges, is_bi=False):
        if is_bi:
            self.graph = nx.DiGraph()
            self.graph.add_nodes_from(nodes)
            self.graph.add_edges_from(edges)
        else:
            self.graph = nx.Graph()
            self.graph.add_nodes_from(nodes)
            self.graph.add_edges_from(edges)

    def max_connected_nodes(self):
        nodes = {}
        for c in nx.connected_components(self.graph):
       # 输出连通子图
            nodes = max(nodes, c, key=lambda x: len(x))
        return nodes
    
    def get_in_degree_zero(self):
        nodes = list(self.graph.nodes)
        zero = [node for node in nodes if self.graph.in_degree(node) == 0]
        return zero
    
    def get_out_degree_zero(self):
        nodes = list(self.graph.nodes)
        zero = [node for node in nodes if self.graph.out_degree(node) == 0]
        return zero
    
    def nerghbor(self, node):
        return list(self.graph.neighbors(node))
