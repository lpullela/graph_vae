import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import stats

# Randomly generate connected graph
def generate_connected_graph(num_nodes, edge_prob):
    while True:
        graph = nx.erdos_renyi_graph(num_nodes, edge_prob)
        
        if nx.is_connected(graph):
            return graph

# Simplified similarity functions for the purpose of testing

def simple_edge_similarity(adj, adj_recon, matching_features):
    """A simple edge similarity function based on adjacency."""
    return np.dot(adj.flatten(), adj_recon.flatten())

def simple_binned_similarity(adj, adj_recon, matching_features):
    """A simple binned similarity function based on degree."""
    degrees = np.sum(adj, axis=1)
    binned_degrees = np.digitize(degrees, bins=np.arange(0, np.max(degrees) + 1))
    return np.mean(binned_degrees)

# currently broken
def simple_page_rank_similarity(adj, adj_recon, matching_features):
    """A simple PageRank-based similarity function."""
    G = nx.from_numpy_array(adj)
    G_recon = nx.from_numpy_array(adj_recon)
    pr = nx.pagerank(G)
    pr_recon = nx.pagerank(G_recon)
    return np.mean([pr[i] * pr_recon[i] for i in range(len(pr))])

def analyze_similarity_performance(num_graphs=100, num_nodes=20, edge_prob_range=(0.1, 0.5)):
    """
    Comprehensive analysis of similarity function performance across different graph characteristics.
    """
    sim_funcs = {
        'simple_edge': simple_edge_similarity,
        'simple_binned': simple_binned_similarity,
        'simple_page_rank': simple_page_rank_similarity
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
        
        matching_features = np.random.rand(num_nodes, 10)
        
        for func_name, sim_func in sim_funcs.items():
            similarity = sim_func(adj1, adj2, matching_features)
            
            metrics = {
                'degree_correlation': stats.pearsonr(
                    [d for n, d in graph1.degree()], 
                    [d for n, d in graph2.degree()]
                )[0],
                'clustering_correlation': stats.pearsonr(
                    list(nx.clustering(graph1).values()), 
                    list(nx.clustering(graph2).values())
                )[0],
                'density_correlation': abs(nx.density(graph1) - nx.density(graph2)),
                'avg_path_length_correlation': abs(
                    nx.average_shortest_path_length(graph1) - 
                    nx.average_shortest_path_length(graph2)
                )
            }
            
            for metric, value in metrics.items():
                performance_metrics[func_name][metric].append((similarity, value))
    
    return performance_metrics

def visualize_similarity_performance(performance_metrics):
    """
    Create visualization to show how similarity functions correlate with graph characteristics.
    """
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