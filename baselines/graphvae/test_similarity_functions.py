import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict

# Randomly generate connected graph
def generate_connected_graph(num_nodes, edge_prob):
    while True:
        graph = nx.erdos_renyi_graph(num_nodes, edge_prob)
        
        if nx.is_connected(graph):
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
def simple_page_rank_similarity(adj, adj_recon, matching_features):
    G = nx.from_numpy_array(adj)
    G_recon = nx.from_numpy_array(adj_recon)
    pr = nx.pagerank(G)
    pr_recon = nx.pagerank(G_recon)
    return np.mean([pr[i] * pr_recon[i] for i in range(len(pr))])

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

def analyze_similarity_performance(num_graphs=400, num_nodes=10, edge_prob_range=(0.2, 0.8)):
    sim_funcs = {
        'Original': simple_edge_similarity,
        'Binned': simple_binned_similarity_binning_method
        # 'simple_binned': simple_binned_similarity,
        # 'simple_page_rank': simple_page_rank_similarity
    }
    
    performance_metrics = {func: {
        'degree_correlation': [],
        'clustering_correlation': [],
        'density_correlation': [],
        'avg_path_length_correlation': []
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
                )
            }
            
            for metric, value in metrics.items():
                performance_metrics[func_name][metric].append((similarity, value))
    
    return performance_metrics

def visualize_similarity_performance(performance_metrics):
    plt.figure(figsize=(15, 10))
    metrics = ['degree_correlation', 'clustering_correlation', 
               'density_correlation', 'avg_path_length_correlation']
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
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