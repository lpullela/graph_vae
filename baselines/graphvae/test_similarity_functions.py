import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict

def generate_connected_graph(num_nodes, edge_prob, graph_type='erdos_renyi'):
    """
        Supported types:
        - 'erdos_renyi': Erdős–Rényi random graph
        - 'barabasi_albert': Scale-free network using preferential attachment
        - 'watts_strogatz': Small-world network
        - 'random_geometric': Geometric random graph
    """
    max_attempts = 100
    
    for _ in range(max_attempts):
        if graph_type == 'erdos_renyi':
            graph = nx.erdos_renyi_graph(num_nodes, edge_prob)
        
        elif graph_type == 'barabasi_albert':
            # edge_prob here represents the number of edges to attach from a new node to existing nodes, can be tuned
            m = max(1, int(edge_prob * num_nodes))
            graph = nx.barabasi_albert_graph(num_nodes, m)
        
        elif graph_type == 'watts_strogatz':
            # edge_prob here represents the probability of rewiring
            k = max(2, int(edge_prob * num_nodes))
            graph = nx.watts_strogatz_graph(num_nodes, k, edge_prob)
        
        elif graph_type == 'random_geometric':
            # edge_prob here represents the radius for connecting points
            graph = nx.random_geometric_graph(num_nodes, edge_prob)
        
        else:
            raise ValueError(f"Unsupported graph type: {graph_type}")
        
        if nx.is_connected(graph):
            return graph
    
    graph = nx.erdos_renyi_graph(num_nodes, edge_prob)
    graph = nx.Graph(graph)
    
    if not nx.is_connected(graph):
        mst = nx.minimum_spanning_tree(nx.complete_graph(graph.nodes()))
        graph.add_edges_from(mst.edges())
    
    return graph

# Simplified similarity functions for the purpose of testing
def simple_edge_similarity(adj, adj_recon, matching_features, matching_features_recon):

    S = np.zeros((adj.shape[0], adj.shape[0], adj.shape[0], adj.shape[0]))
    for i in range(adj.shape[0]):
        for j in range(adj.shape[0]):
            if i == j:
                for a in range(adj.shape[0]):
                    S[i, i, a, a] = adj[i, i] * adj_recon[a, a] * \
                                    np.dot(matching_features[i], matching_features_recon[a])
            else:
                for a in range(adj.shape[0]):
                    for b in range(adj.shape[0]):
                        if b == a:
                            continue
                        S[i, j, a, b] = adj[i, j] * adj[i, i] * adj[j, j] * \
                                        adj_recon[a, b] * adj_recon[a, a] * adj_recon[b, b]
    return np.mean(S)


def simple_binned_similarity(adj, adj_recon, matching_features):
    degrees = np.sum(adj, axis=1)
    binned_degrees = np.digitize(degrees, bins=np.arange(0, np.max(degrees) + 1))
    return np.mean(binned_degrees)

# currently broken
def simple_page_rank_similarity_binning_method(adj, adj_recon, matching_features, matching_features_recon):
    def compute_page_rank(A):
        G = nx.from_numpy_array(A)
        rank_dict = nx.pagerank(G)
        return rank_dict

    rank_dict_adj = compute_page_rank(adj)
    rank_dict_adj_recon = compute_page_rank(adj_recon)

    # Bin nodes based on their PageRank values
    def binning_page_rank(rank_dict, num_bins=5):
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

    adj_bins = binning_page_rank(rank_dict_adj)
    adj_recon_bins = binning_page_rank(rank_dict_adj_recon)

    S = np.zeros((adj.shape[0], adj.shape[0], adj.shape[0], adj.shape[0]))

    for rank_bin in adj_bins:
        if rank_bin in adj_recon_bins:
            adj_nodes = adj_bins[rank_bin]
            adj_recon_nodes = adj_recon_bins[rank_bin]

            for i in adj_nodes:
                for j in adj_nodes:
                    if i == j:
                        # Case when i == j
                        for a in adj_recon_nodes:
                            S[i, i, a, a] = adj[i, i] * adj_recon[a, a] * \
                                            np.dot(matching_features[i], matching_features_recon[a])
                    else:
                        # Case when i != j
                        for a in adj_recon_nodes:
                            for b in adj_recon_nodes:
                                if b == a:
                                    continue
                                S[i, j, a, b] = (
                                    adj[i, j] * adj[i, i] * adj[j, j] *
                                    adj_recon[a, b] * adj_recon[a, a] * adj_recon[b, b]
                                )

    return np.mean(S)

# Community-detection-based similarity function
def simple_community_similarity(adj, adj_recon, matching_features, matching_features_recon):
    S = np.zeros((adj.shape[0], adj.shape[0], adj.shape[0], adj.shape[0]))

    # numpy bug fix
    if isinstance(adj, torch.Tensor):
        adj = adj.cpu().numpy()
    if isinstance(adj_recon, torch.Tensor):
        adj_recon = adj_recon.cpu().numpy()

    communities = find_communities(adj)
    communities_recon = find_communities(adj_recon)

    for comm_id, nodes in communities.items():
        comm_size = len(nodes)
        for recon_comm_id, recon_nodes in communities_recon.items():
            # Allow small discrepancies in community size
            if abs(len(recon_nodes) - comm_size) <= 2:
                for i in nodes:
                    for j in nodes:
                        if i == j:
                            for a in recon_nodes:
                                S[i, i, a, a] = adj[i, i] * adj_recon[a, a] * \
                                                np.dot(matching_features[i], matching_features_recon[a])
                        else:
                            for a in recon_nodes:
                                for b in recon_nodes:
                                    if b == a:
                                        continue
                                    S[i, j, a, b] = (
                                        adj[i, j] * adj[i, i] * adj[j, j] *
                                        adj_recon[a, b] * adj_recon[a, a] * adj_recon[b, b]
                                    )

    return np.mean(S)

# Helper function to find communities
def find_communities(adj):
    n = adj.shape[0]
    communities = {i: [i] for i in range(n)}
    
    def calculate_modularity(adj, communities):
        m = np.sum(adj) / 2
        Q = 0
        for nodes in communities.values():
            for i in nodes:
                for j in nodes:
                    A_ij = adj[i, j]
                    k_i = np.sum(adj[i])
                    k_j = np.sum(adj[j])
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

def simple_binned_similarity_binning_method(adj, adj_recon, matching_features, matching_features_recon):
    # Bin nodes by degree
    def bin_nodes_by_degree(adj, binary=True, threshold=0.8):
        if not binary: 
            binary_adj = (adj > threshold).astype(float)
        else: 
            binary_adj = adj
        degrees = np.sum(binary_adj, axis=1)
        bins = defaultdict(list)
        for node, degree in enumerate(degrees):
            degree_int = int(degree)
            for offset in [-1, 0, 1]:
                bin_key = degree_int + offset
                if bin_key >= 0:
                    bins[bin_key].append(node)
        return bins

    adj_bins = bin_nodes_by_degree(adj)
    adj_recon_bins = bin_nodes_by_degree(adj_recon, binary=False)

    S = np.zeros((adj.shape[0], adj.shape[0], adj.shape[0], adj.shape[0]))

    # Only compute similarities within matched degree bins
    for degree in adj_bins:
        if degree in adj_recon_bins:
            adj_nodes = adj_bins[degree]
            adj_recon_nodes = adj_recon_bins[degree]

            for i in adj_nodes:
                for j in adj_nodes:
                    if i == j:
                        # Case when i == j
                        for a in adj_recon_nodes:
                            S[i, i, a, a] = adj[i, i] * adj_recon[a, a] * \
                                            np.dot(matching_features[i], matching_features_recon[a])
                    else:
                        # Case when i != j
                        for a in adj_recon_nodes:
                            for b in adj_recon_nodes:
                                if b == a:
                                    continue
                                S[i, j, a, b] = (
                                    adj[i, j] * adj[i, i] * adj[j, j] *
                                    adj_recon[a, b] * adj_recon[a, a] * adj_recon[b, b]
                                )

    return np.mean(S)

def analyze_similarity_performance(num_graphs=200, num_nodes=10, edge_prob_range=(0.2, 0.8)):
    sim_funcs = {
        'Original': simple_edge_similarity,
        'Binned': simple_binned_similarity_binning_method,
        'PageRank': simple_page_rank_similarity_binning_method,
        'Community': simple_community_similarity
    }
    
    performance_metrics = {func: {
        'degree_correlation': [],
        'clustering_correlation': [],
        'density_correlation': [],
        'avg_path_length_correlation': [],
        'diameter_correlation': []
    } for func in sim_funcs}
    
    for _ in range(num_graphs):
        edge_prob1 = np.random.uniform(*edge_prob_range)
        edge_prob2 = np.random.uniform(*edge_prob_range)
        
        graph1 = generate_connected_graph(num_nodes, edge_prob1)
        graph2 = generate_connected_graph(num_nodes, edge_prob2)
        
        adj1 = nx.to_numpy_array(graph1)
        adj2 = nx.to_numpy_array(graph2)

        np.fill_diagonal(adj1, 1)
        np.fill_diagonal(adj2, 1)
        
        matching_features1 = np.random.rand(num_nodes, 10)
        # matching_features2 = np.random.rand(num_nodes, 10)
        matching_features2 = matching_features1
        
        for func_name, sim_func in sim_funcs.items():
            similarity = sim_func(adj1, adj1, matching_features1, matching_features1)
            
            metrics = {
                # Degree-degree correlation
                # Higher means more similar degree distribution
                'degree_correlation': stats.pearsonr(
                    [d for n, d in graph1.degree()], 
                    [d for n, d in graph2.degree()]
                )[0],
                # Clustering coefficient correlation
                # Higher means more similar clustering coefficient
                'clustering_correlation': stats.pearsonr(
                    list(nx.clustering(graph1).values()), 
                    list(nx.clustering(graph2).values())
                )[0],
                # Absolute difference in graph density (edges / max edges)
                # Smaller absolute value means more similar density
                'density_correlation': abs(nx.density(graph1) - nx.density(graph2)),
                # Absolute difference in APL
                # Smaller absolute value means more similar APL
                'avg_path_length_correlation': abs(
                    nx.average_shortest_path_length(graph1) - 
                    nx.average_shortest_path_length(graph2)
                ),
                'diameter_correlation': abs(
                    nx.diameter(graph1) - nx.diameter(graph2)
                )
            }
            
            for metric, value in metrics.items():
                performance_metrics[func_name][metric].append((similarity, value))
    
    return performance_metrics

def visualize_similarity_performance(performance_metrics):
    plt.figure(figsize=(15, 10))
    metrics = ['degree_correlation', 'clustering_correlation', 
               'density_correlation', 'avg_path_length_correlation', 'diameter_correlation']
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 3, i)
        for func_name, data in performance_metrics.items():
            similarities, correlations = zip(*data[metric])
            plt.scatter(similarities, correlations, label=func_name, alpha=0.7)
        
        plt.title(f'Performance on {metric.replace("_", " ").title()}')
        plt.xlabel('Similarity Score')
        plt.ylabel('Graph Characteristic Correlation')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    performance_metrics = analyze_similarity_performance()
    visualize_similarity_performance(performance_metrics)
    
    for func_name, metrics in performance_metrics.items():
        print(f"\nPerformance Summary for {func_name}:")
        for metric, values in metrics.items():
            similarities, correlations = zip(*values)
            print(f"  {metric.replace('_', ' ').title()}:")
            print(f"    Average Correlation: {np.mean(correlations):.4f}")
            print(f"    Correlation Variance: {np.var(correlations):.4f}")

if __name__ == "__main__":
    main()