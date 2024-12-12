import torch 
import numpy as np
import networkx as nx
from collections import defaultdict
import time
import copy
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
import math

class SimilarityFunctions(): 

    def __init__(self,similarity_function, max_nodes): 
        self.sim_func_name = similarity_function
        self.max_num_nodes = max_nodes
        self._timer_start = None  
        self._elapsed_time = 0.0  
        self.time_stamps = []

    def _start_timer(self):
        if self._timer_start is None:
            self._timer_start = time.time()

    def _stop_timer(self):
        if self._timer_start is not None:
            self._elapsed_time += time.time() - self._timer_start
            self.time_stamps.append(time.time() - self._timer_start)
            self._timer_start = None

    def get_elapsed_time(self):
        return self._elapsed_time

    # original brute force solution
    def edge_similarity_matrix(self, adj, adj_recon, matching_features,
                matching_features_recon, sim_func):
        self._start_timer()
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
        self._stop_timer()
        return S
    
    def bin_nodes_by_degree(self, adj, binary=True, threshold=0.5):
        
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

        return degrees, bins

    # modified similarity function with binning method
    def edge_similarity_matrix_binning_method(self, adj, adj_recon, matching_features, matching_features_recon, sim_func):

        self._start_timer()
        adj_degrees, adj_bins = self.bin_nodes_by_degree(adj)
        adj_recon_degrees, adj_recon_bins = self.bin_nodes_by_degree(adj_recon, binary=False)

        # # for testing purposes in bin density
        # n = adj.shape[0]  
        # k = len(adj_recon_bins.keys())  
        # nk_ratio = n / k
        # with open("baselines/graphvae/results/nk_ratios.txt", "a") as file:  
        #     file.write(f"{nk_ratio}\n")
        # # can delete the above or comment

        S = torch.zeros(self.max_num_nodes, self.max_num_nodes, self.max_num_nodes, self.max_num_nodes)

        # only compute similarities within matched degree bins
        for degree in adj_bins:
            if degree in adj_recon_bins:
                adj_nodes = adj_bins[degree]
                adj_recon_nodes = adj_recon_bins[degree]

                for i in adj_nodes:
                    for j in range(self.max_num_nodes):
                        if i == j:
                            # case when i == j
                            for a in adj_recon_nodes: # these should prolly be max
                                S[i, i, a, a] = adj[i, i] * adj_recon[a, a] * sim_func(matching_features[i], matching_features_recon[a])
                        else:
                            deg_j = adj_degrees[j]
                            if deg_j in adj_recon_bins:
                                possible_j_matches = adj_recon_bins[deg_j]
                                # case when i != j
                                for a in adj_recon_nodes:
                                    for b in possible_j_matches:
                                        if b == a:
                                            continue
                                        S[i, j, a, b] = (
                                            adj[i, j] * adj[i, i] * adj[j, j]
                                            * adj_recon[a, b] * adj_recon[a, a] * adj_recon[b, b]
                                        )

        self._stop_timer()
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

    def binning_page_rank(self, rank_dict):
        sorted_ranks_list = sorted(rank_dict.items(), key=lambda x: x[1])
        sorted_ranks = {node: rank + 1 for rank, (node, _) in enumerate(sorted_ranks_list)}
        binned_dict = {}

        for node, rank in sorted_ranks.items():
            for key_rank in range(max(1,rank - 2), min(len(rank_dict) + 1, rank + 3)):  # Keys are based on ranks within +-2
                if not key_rank in binned_dict: 
                    binned_dict[key_rank] = []
                binned_dict[key_rank].append(node)

        return sorted_ranks, binned_dict
    
    
    def edge_similarity_matrix_page_rank_method(self, adj, adj_recon, matching_features, matching_features_recon, sim_func):
        self._start_timer()
        rank_dict_adj = self.compute_page_rank(adj)
        rank_dict_adj_recon = self.compute_page_rank(adj_recon)

        sorted_ranks_adj, adj_bins = self.binning_page_rank(rank_dict_adj)
        sorted_ranks_adj_recon, adj_recon_bins = self.binning_page_rank(rank_dict_adj_recon)

        S = torch.zeros(self.max_num_nodes, self.max_num_nodes, self.max_num_nodes, self.max_num_nodes)

        for rank_bin in adj_bins:
            adj_nodes = adj_bins[rank_bin]
            adj_recon_nodes = adj_recon_bins[rank_bin]

            for i in adj_nodes:
                for j in range(self.max_num_nodes):
                    if i == j:
                        # Case when i == j
                        for a in adj_recon_nodes:
                            S[i, i, a, a] = adj[i, i] * adj_recon[a, a] * sim_func(matching_features[i], matching_features_recon[a])
                    else:
                        rank_j = sorted_ranks_adj[j]
                        possible_j_matches = adj_recon_bins[rank_j]
                        # Case when i != j
                        for a in adj_recon_nodes:
                            for b in possible_j_matches:
                                if b == a:
                                    continue
                                S[i, j, a, b] = (
                                    adj[i, j] * adj[i, i] * adj[j, j]
                                    * adj_recon[a, b] * adj_recon[a, a] * adj_recon[b, b]
                                )

        self._stop_timer()
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
        self._start_timer()
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

        self._stop_timer()
        return S
    
    # louvain communities
    def find_louvain_communities(self, adj, seed=42): # randomness in initialization of singleton communities on first it.
        adj_matrix = adj.cpu().numpy()
        graph = nx.from_numpy_array(adj_matrix)
        return nx.community.louvain_communities(graph, seed=42)
    
    def normalize_dict_values(self, modularities):
        values = np.array(list(modularities.values()))
        mean = np.mean(values)
        std = np.std(values)
        normalized = {k: (v - mean) / std for k, v in modularities.items()}
        return normalized
    
    def get_comm_modularity(self, adj, nodes, binary=False, threshold=0.5): 
        d_adj = copy.deepcopy(adj)
        if not binary: 
            d_adj = (d_adj > threshold).float()

        m = torch.sum(d_adj).item() // 2
        degrees = torch.sum(d_adj, dim=1)

        total = 0
        for i in nodes: 
            for j in nodes:  
                total += d_adj[i][j] - (degrees[i] * degrees[j]) / (2 * m)

        return total / (2 * m)  # calculation for modularity within community

    # louvaine community detection
    def edge_similarity_matrix_louvain_method(self, adj, adj_recon, matching_features, matching_features_recon, sim_func):
        self._start_timer()
        S = torch.zeros(self.max_num_nodes, self.max_num_nodes, self.max_num_nodes, self.max_num_nodes)
        
        # numpy bug fix
        if isinstance(adj, torch.Tensor):
            adj = adj.cpu()
        if isinstance(adj_recon, torch.Tensor):
            adj_recon = adj_recon.cpu()

        communities = self.find_louvain_communities(adj)
        communities_recon = self.find_louvain_communities(adj_recon)

        modularities_adj = {comm_id: self.get_comm_modularity(adj, nodes) for comm_id, nodes in enumerate(communities)}
        modularities_adj_recon = {comm_id: self.get_comm_modularity(adj_recon, nodes) for comm_id, nodes in enumerate(communities_recon)}

        modularities_adj = self.normalize_dict_values(modularities_adj)
        modularities_adj_recon = self.normalize_dict_values(modularities_adj_recon)
        
        mapping = {}
        for comm_id_adj, value_adj in modularities_adj.items():
            closest_comm_id = min(
                modularities_adj_recon.keys(),
                key=lambda comm_id_recon: abs(modularities_adj_recon[comm_id_recon] - value_adj)
            )
            mapping[comm_id_adj] = closest_comm_id
            # # Optionally remove the matched `comm_id_recon` to prevent duplicate matches
            # modularities_adj_recon.pop(closest_comm_id)

        for comm_id, nodes in enumerate(communities):
            recon_nodes = communities_recon[mapping[comm_id]]
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

        self._stop_timer()
        return S
    
    def find_spectral_communities(self, adj, num_clusters=5): # start with 5 clusters to mirror page rank
        adj_matrix = adj.cpu().numpy()
        graph = nx.from_numpy_array(adj_matrix)
        L = nx.normalized_laplacian_matrix(graph).astype(float)
        eigvals, eigvecs = eigsh(L, k=num_clusters, which='SM')
        
        # using k means to cluster nodes based on eigenvalue strongness
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        labels = kmeans.fit_predict(eigvecs)
        
        # group similar nodes together by their label
        communities = [set() for _ in range(num_clusters)]
        for node, label in zip(graph.nodes(), labels):
            communities[label].add(node)
        
        return communities
    
    def edge_similarity_matrix_spectral_method(self, adj, adj_recon, matching_features, matching_features_recon, sim_func):
        self._start_timer()
        S = torch.zeros(self.max_num_nodes, self.max_num_nodes, self.max_num_nodes, self.max_num_nodes)
        
        # numpy bug fix
        if isinstance(adj, torch.Tensor):
            adj = adj.cpu()
        if isinstance(adj_recon, torch.Tensor):
            adj_recon = adj_recon.cpu()

        communities = self.find_spectral_communities(adj)
        communities_recon = self.find_spectral_communities(adj_recon)

        # i think we should keep the louvain error feature of modularity. it's a good metric to see how communal a community is 
        modularities_adj = {comm_id: self.get_comm_modularity(adj, nodes) for comm_id, nodes in enumerate(communities)}
        modularities_adj_recon = {comm_id: self.get_comm_modularity(adj_recon, nodes) for comm_id, nodes in enumerate(communities_recon)}

        modularities_adj = self.normalize_dict_values(modularities_adj)
        modularities_adj_recon = self.normalize_dict_values(modularities_adj_recon)
        
        mapping = {}
        for comm_id_adj, value_adj in modularities_adj.items():
            closest_comm_id = min(
                modularities_adj_recon.keys(),
                key=lambda comm_id_recon: abs(modularities_adj_recon[comm_id_recon] - value_adj)
            )
            mapping[comm_id_adj] = closest_comm_id
            # # Optionally remove the matched `comm_id_recon` to prevent duplicate matches
            # modularities_adj_recon.pop(closest_comm_id)

        for comm_id, nodes in enumerate(communities):
            recon_nodes = communities_recon[mapping[comm_id]]
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

        self._stop_timer()
        return S
    
    def compute_betweenness_centrality(self, adjacency_matrix):
        if isinstance(adjacency_matrix, torch.Tensor):
            adjacency_matrix = adjacency_matrix.cpu().numpy()  # Convert to NumPy

        G = nx.from_numpy_array(adjacency_matrix)
        centrality_dict = nx.betweenness_centrality(G, normalized=True)
        return centrality_dict
    
    def edge_similarity_matrix_betweeness_method(self, adj, adj_recon, matching_features, matching_features_recon, sim_func):
        self._start_timer()
        rank_dict_adj = self.compute_betweenness_centrality(adj)
        rank_dict_adj_recon = self.compute_betweenness_centrality(adj_recon)
        nodes_per_bin = 5
        
        sorted_ranks_adj, adj_bins = self.binning_page_rank(rank_dict_adj)
        sorted_ranks_adj_recon, adj_recon_bins = self.binning_page_rank(rank_dict_adj_recon)

        S = torch.zeros(self.max_num_nodes, self.max_num_nodes, self.max_num_nodes, self.max_num_nodes)

        for rank_bin in adj_bins:
            adj_nodes = adj_bins[rank_bin]
            adj_recon_nodes = adj_recon_bins.get(rank_bin, [])  # Default to an empty list if rank_bin not in adj_recon_bins

            for i in adj_nodes:
                for j in range(self.max_num_nodes):
                    if i == j:
                        # Case when i == j
                        for a in adj_recon_nodes:
                            S[i, i, a, a] = adj[i, i] * adj_recon[a, a] * sim_func(matching_features[i], matching_features_recon[a])
                    else:
                        rank_j = sorted_ranks_adj[j]
                        possible_j_matches = adj_recon_bins[rank_j]
                        # Case when i != j
                        for a in adj_recon_nodes:
                            for b in possible_j_matches:
                                if b == a:
                                    continue
                                S[i, j, a, b] = (
                                    adj[i, j] * adj[i, i] * adj[j, j]
                                    * adj_recon[a, b] * adj_recon[a, a] * adj_recon[b, b]
                                )

        self._stop_timer()
        return S





   