import torch 
import numpy as np
import networkx as nx
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
            #bins[int(degree.item())].append(node)
            degree_int = int(degree.item())
            for offset in [-1, 0, 1]:
                # instead of mapping to only one possible degree, threshold to +/- 1
                # mapping nodes to a range of degrees is producing better results than mapping to one deg
                bin_key = degree_int + offset
                if bin_key >= 0:
                    bins[bin_key].append(node)

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
    
    # another idea is maybe trying to use a page rank algo? 
    # we can rank all the nodes by 'influence' and then bin them
    # and then instead of binning on degree we can bin on rank
    # just an idea

    def compute_page_rank(self, A):
        # use the nx implementation to compute the ranks of the nodes
        # ranks contain info about each nodes 'influence'
        G = nx.from_numpy_array(A.numpy() if isinstance(A, torch.Tensor) else A)
        rank_dict = nx.pagerank(G)
        return rank_dict

    def binning_page_rank(self, rank_dict, num_bins=5):
        # bin nodes based on their page rnak values.
        # sort nodes by page rank

        sorted_ranks = sorted(rank_dict.items(), key=lambda x: x[1])
        binned_dict = {}
        bin_size = len(rank_dict) // num_bins

        for bin_index in range(num_bins):
            start_idx = bin_index * bin_size
            end_idx = (bin_index + 1) * bin_size if bin_index < num_bins - 1 else len(rank_dict)

            # Collect nodes in the bin
            binned_nodes = [node for node, _ in sorted_ranks[start_idx:end_idx]]
            binned_dict[bin_index] = binned_nodes

        return binned_dict
    
    
    def edge_similarity_matrix_page_rank_method(self, adj, adj_recon, matching_features, matching_features_recon, sim_func):
        rank_dict_adj = self.compute_page_rank(adj)
        rank_dict_adj_recon = self.compute_page_rank(adj_recon)
        
        adj_bins = self.binning_page_rank(rank_dict_adj)
        adj_recon_bins = self.binning_page_rank(rank_dict_adj_recon)

        S = torch.zeros(self.max_num_nodes, self.max_num_nodes, self.max_num_nodes, self.max_num_nodes)

        for rank_bin in adj_bins:
            if rank_bin in adj_recon_bins:
                adj_nodes = adj_bins[rank_bin]
                adj_recon_nodes = adj_recon_bins[rank_bin]

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
    


   