import torch 
from collections import defaultdict

class SimilarityFunctions(): 

    def __init__(self,similarity_function, max_nodes): 
        self.sim_func_name = similarity_function
        self.max_num_nodes = max_nodes

    # original brute force solution
    def edge_similarity_matrix(self, adj, adj_recon, matching_features,
                matching_features_recon, sim_func):
        S = torch.zeros(self.max_num_nodes, self.max_num_nodes,
                        self.max_num_nodes, self.max_num_nodes)
        for i in range(self.max_num_nodes):
            for j in range(self.max_num_nodes):
                if i == j:
                    for a in range(self.max_num_nodes):
                        S[i, i, a, a] = adj[i, i] * adj_recon[a, a] * \
                                        sim_func(matching_features[i], matching_features_recon[a])
                        # with feature not implemented
                        # if input_features is not None:
                else:
                    for a in range(self.max_num_nodes):
                        for b in range(self.max_num_nodes):
                            if b == a:
                                continue
                            S[i, j, a, b] = adj[i, j] * adj[i, i] * adj[j, j] * \
                                            adj_recon[a, b] * adj_recon[a, a] * adj_recon[b, b]
        return S
    
    def bin_nodes_by_degree(self, adj, binary=True, threshold=0.8):

        if not binary: 
            binary_adj = (adj > threshold).float()
        else: 
            binary_adj = adj

        degrees = torch.sum(binary_adj, dim=1)

        bins = defaultdict(list)
        for node, degree in enumerate(degrees):
            bins[int(degree.item())].append(node)

        return bins

    # modified similarity function with binning method
    def edge_similarity_matrix_binning_method(self, adj, adj_recon, matching_features, matching_features_recon, sim_func):

        adj_bins = self.bin_nodes_by_degree(adj)
        adj_recon_bins = self.bin_nodes_by_degree(adj_recon, binary=False)

        S = torch.zeros(self.max_num_nodes, self.max_num_nodes, self.max_num_nodes, self.max_num_nodes)

        # only compute similarities within matched degree bins
        for degree in adj_bins:
            if degree in adj_recon_bins:
                adj_nodes = adj_bins[degree]
                adj_recon_nodes = adj_recon_bins[degree]

                for i in adj_nodes:
                    for j in adj_nodes:
                        if i == j:
                            # case when i == j
                            for a in adj_recon_nodes:
                                S[i, i, a, a] = adj[i, i] * adj_recon[a, a] * sim_func(matching_features[i], matching_features_recon[a])
                        else:
                            # case when i != j
                            for a in adj_recon_nodes:
                                for b in adj_recon_nodes:
                                    if b == a:
                                        continue
                                    S[i, j, a, b] = (
                                        adj[i, j] * adj[i, i] * adj[j, j]
                                        * adj_recon[a, b] * adj_recon[a, a] * adj_recon[b, b]
                                    )

        return S