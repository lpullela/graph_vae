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
    
    #Girvan-Newman community detection
    def find_communities(self, adj):
        n = adj.shape[0]
        communities = {i: [i] for i in range(n)}
        
        def calculate_modularity(adj, communities):
            m = torch.sum(adj) / 2
            Q = 0
            for nodes in communities.values():
                for i in nodes:
                    for j in nodes:
                        A_ij = adj[i, j]
                        k_i = torch.sum(adj[i])
                        k_j = torch.sum(adj[j])
                        Q += (A_ij - (k_i * k_j) / (2 * m))
            return Q / (2 * m)

        def merge_communities(communities, adj):
            best_Q = calculate_modularity(adj, communities)
            best_merge = None
            keys = list(communities.keys())
            
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    temp_communities = communities.copy()
                    temp_communities[keys[i]] = communities[keys[i]] + communities[keys[j]]
                    del temp_communities[keys[j]]
                    
                    Q = calculate_modularity(adj, temp_communities)
                    if Q > best_Q:
                        best_Q = Q
                        best_merge = (keys[i], keys[j])
            
            return best_merge, best_Q

        while True:
            best_merge, best_Q = merge_communities(communities, adj)
            if best_merge is None:
                break
            
            i, j = best_merge
            communities[i].extend(communities[j])
            del communities[j]
        
        return communities

    # Community-detection-based similarity function
    # Basic idea: take advantage of repetition in long, chain-like structures (e.g. enzymes)
    # Only compare nodes if they are in the same community
    def edge_similarity_matrix_community_method(self, adj, adj_recon, matching_features, matching_features_recon, sim_func):
        S = torch.zeros(self.max_num_nodes, self.max_num_nodes, self.max_num_nodes, self.max_num_nodes)
        
        # numpy bug fix
        if isinstance(adj, torch.Tensor):
            adj = adj.cpu()
        if isinstance(adj_recon, torch.Tensor):
            adj_recon = adj_recon.cpu()

        communities = self.find_communities(adj)
        communities_recon = self.find_communities(adj_recon)

        for comm_id, nodes in communities.items():
            comm_size = len(nodes)
            for recon_comm_id, recon_nodes in communities_recon.items():
                # allow small discrepancies in community size
                if abs(len(recon_nodes) - comm_size) <= 2:
                    for i in nodes:
                        for j in nodes:
                            if i == j:
                                for a in recon_nodes:
                                    S[i, i, a, a] = adj[i, i] * adj_recon[a, a] * sim_func(matching_features[i], matching_features_recon[a])
                            else:
                                for a in recon_nodes:
                                    for b in recon_nodes:
                                        if b == a:
                                            continue
                                        S[i, j, a, b] = (
                                            adj[i, j] * adj[i, i] * adj[j, j] * adj_recon[a, b] * adj_recon[a, a] * adj_recon[b, b]
                                        )

        return S



   