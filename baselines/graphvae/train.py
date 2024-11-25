
import argparse
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
from random import shuffle
import torch
import torch.nn as nn
import torch.nn.init as init 
from torch.autograd import Variable 
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from similarity_functions import SimilarityFunctions
import time
import matplotlib.pyplot as plt
from datetime import datetime

#import data
# from baselines.graphvae.model import GraphVAE
# from baselines.graphvae.data import GraphAdjSampler
from model import GraphVAE
from data import GraphAdjSampler

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

LR_milestones = [500, 1000]

# load ENZYMES and PROTEIN and DD dataset
def Graph_load_batch(min_num_nodes = 20, max_num_nodes = 1000, name = 'ENZYMES',node_attributes = True,graph_labels=True):
    '''
    load many graphs, e.g. enzymes
    :return: a list of graphs
    '''
    print('Loading graph dataset: '+str(name))
    G = nx.Graph()
    # load data
    path = 'dataset/'+name+'/'
    data_adj = np.loadtxt(path+name+'_A.txt', delimiter=',').astype(int)
    if node_attributes:
        data_node_att = np.loadtxt(path+name+'_node_attributes.txt', delimiter=',')
    data_node_label = np.loadtxt(path+name+'_node_labels.txt', delimiter=',').astype(int)
    data_graph_indicator = np.loadtxt(path+name+'_graph_indicator.txt', delimiter=',').astype(int)
    if graph_labels:
        data_graph_labels = np.loadtxt(path+name+'_graph_labels.txt', delimiter=',').astype(int)


    data_tuple = list(map(tuple, data_adj))
    # print(len(data_tuple))
    # print(data_tuple[0])

    # add edges
    G.add_edges_from(data_tuple)
    # add node attributes
    for i in range(data_node_label.shape[0]):
        if node_attributes:
            G.add_node(i+1, feature = data_node_att[i])
        G.add_node(i+1, label = data_node_label[i])
    G.remove_nodes_from(list(nx.isolates(G)))

    # print(G.number_of_nodes())
    # print(G.number_of_edges())

    # split into graphs
    graph_num = data_graph_indicator.max()
    node_list = np.arange(data_graph_indicator.shape[0])+1
    graphs = []
    max_nodes = 0
    for i in range(graph_num):
        # find the nodes for each graph
        nodes = node_list[data_graph_indicator==i+1]
        G_sub = G.subgraph(nodes)
        if graph_labels:
            G_sub.graph['label'] = data_graph_labels[i]
        # print('nodes', G_sub.number_of_nodes())
        # print('edges', G_sub.number_of_edges())
        # print('label', G_sub.graph)
        if G_sub.number_of_nodes()>=min_num_nodes and G_sub.number_of_nodes()<=max_num_nodes:
            graphs.append(G_sub)
            if G_sub.number_of_nodes() > max_nodes:
                max_nodes = G_sub.number_of_nodes()
            # print(G_sub.number_of_nodes(), 'i', i)
    # print('Graph dataset name: {}, total graph num: {}'.format(name, len(graphs)))
    # logging.warning('Graphs loaded, total num: {}'.format(len(graphs)))
    print('Loaded')
    return graphs

def build_model(args, max_num_nodes):
    out_dim = max_num_nodes * (max_num_nodes + 1) // 2
    if args.feature_type == 'id':
        input_dim = max_num_nodes
    elif args.feature_type == 'deg':
        input_dim = 1
    elif args.feature_type == 'struct':
        input_dim = 2
    model = GraphVAE(input_dim, 64, 256, max_num_nodes)
    return model

def save_result(args, test_loss, training_time): 
    if isinstance(test_loss, torch.Tensor):
        test_loss = test_loss.item()  
    
    output_dir = "baselines/graphvae/results"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"test_loss_{args.similarity_function}_{args.epochs}_{args.max_num_nodes}.txt")

    with open(file_path, "w") as f:
        f.write(f"Test Loss: {test_loss}\n")
        f.write(f"Training Time: {training_time}\n")
    
    print(f"Test loss saved to {file_path}")

def save_plot(lst, args): 
    lst = [x.item() if isinstance(x, torch.Tensor) else x for x in lst]

    plt.plot(lst)
    plt.xlabel("Mini batch")
    plt.ylabel("Loss (BCE + KL-Div)")
    plt.title("Training Loss")
    output_dir = "baselines/graphvae/plots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"tr_loss_{args.similarity_function}_{args.epochs}_{args.max_num_nodes}.png"))
    plt.clf()

    array_path = os.path.join(output_dir, f"tr_loss_{args.similarity_function}_{args.epochs}_{args.max_num_nodes}.npy")
    np.save(array_path, lst)

def train(args, dataloader, model):
    # set timer
    start = time.time()

    epoch = 1
    optimizer = optim.Adam(list(model.parameters()), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=LR_milestones, gamma=args.lr)

    model.train()
    model.to(device)
    sim_func_obj = SimilarityFunctions(args.similarity_function, args.max_num_nodes)

    loss_logger = []

    for epoch in range(int(args.epochs)):
        for batch_idx, data in enumerate(dataloader):
            model.zero_grad()
            features = data['features'].float()
            adj_input = data['adj'].float()

            features = Variable(features).to(device)
            adj_input = Variable(adj_input).to(device)
            
            loss = model(features, adj_input, sim_func_obj)
            loss_logger.append(loss)
            print('Epoch: ', epoch, ', Iter: ', batch_idx, ', Loss: ', loss)
            loss.backward()

            optimizer.step()
            scheduler.step()
            break

    end = time.time()
    length = end - start

    save_plot(loss_logger, args)
    print("Training length in seconds: ", length)
    print("Training length in minutes: ", length/60)

    return length

def test(args, dataloader, model, time): 

    test_loss = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            features = data['features'].float()
            adj_input = data['adj'].float()

            features = Variable(features).to(device)
            adj_input = Variable(adj_input).to(device)

            test_loss += model.forward_test(features, adj_input, args)

    print("test loss: ", test_loss)
    # save result (test loss) to results dir 
    save_result(args, test_loss, time)

    return test_loss


def arg_parse():
    parser = argparse.ArgumentParser(description='GraphVAE arguments.')
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument('--dataset', dest='dataset', 
            help='Input dataset.')

    parser.add_argument('--lr', dest='lr', type=float,
            help='Learning rate.')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
            help='Batch size.')
    parser.add_argument('--num_workers', dest='num_workers', type=int,
            help='Number of workers to load data.')
    parser.add_argument('--max_num_nodes', dest='max_num_nodes', type=int,
            help='Predefined maximum number of nodes in train/test graphs. -1 if determined by \
                  training data.')
    parser.add_argument('--feature', dest='feature_type',
            help='Feature used for encoder. Can be: id, deg')
    parser.add_argument('--similarity_function', dest='similarity_function', 
            help='Function used to compute similarity matrix (to find permutations between between A and A_recon)')
    parser.add_argument('--epochs', dest='epochs',
            help='Number of epochs to train data.')

    parser.set_defaults(dataset='grid',
                        feature_type='id',
                        lr=0.001,
                        batch_size=1,
                        num_workers=1,
                        max_num_nodes=-1,
                        similarity_function='original', 
                        epochs=25)
    return parser.parse_args()

def main():
    prog_args = arg_parse()

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(CUDA)
    # print('CUDA', CUDA)
    ### running log

    if prog_args.dataset == 'enzymes':
        graphs= Graph_load_batch(min_num_nodes=10, name='ENZYMES')
        num_graphs_raw = len(graphs)
    elif prog_args.dataset == 'grid':
        graphs = []
        for i in range(2,3):
            for j in range(2,3):
                graphs.append(nx.grid_2d_graph(i,j))
        num_graphs_raw = len(graphs)

    if prog_args.max_num_nodes == -1:
        max_num_nodes = max([graphs[i].number_of_nodes() for i in range(len(graphs))])
    else:
        max_num_nodes = prog_args.max_num_nodes
        # remove graphs with number of nodes greater than max_num_nodes
        graphs = [g for g in graphs if g.number_of_nodes() <= max_num_nodes]

    graphs_len = len(graphs)
    print('Number of graphs removed due to upper-limit of number of nodes: ', 
            num_graphs_raw - graphs_len)
    graphs_test = graphs[int(0.8 * graphs_len):]
    graphs_train = graphs[0:int(0.8*graphs_len)]
    #graphs_train = graphs

    print('total graph num: {}, training set: {}'.format(len(graphs),len(graphs_train)))
    print('max number node: {}'.format(max_num_nodes))

    dataset = GraphAdjSampler(graphs_train, max_num_nodes, features=prog_args.feature_type)
    dataset_test = GraphAdjSampler(graphs_test, max_num_nodes, features=prog_args.feature_type)
    #sample_strategy = torch.utils.data.sampler.WeightedRandomSampler(
    #        [1.0 / len(dataset) for i in range(len(dataset))],
    #        num_samples=prog_args.batch_size, 
    #        replacement=False)
    dataset_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=prog_args.batch_size, 
            num_workers=prog_args.num_workers)
    
    dataset_loader_test = torch.utils.data.DataLoader(
        dataset_test, 
        batch_size=prog_args.batch_size, 
        num_workers=prog_args.num_workers)
    
    model = build_model(prog_args, max_num_nodes).to(device)
    time = train(prog_args, dataset_loader, model)

    test(prog_args, dataset_loader_test, model, time)


if __name__ == '__main__':
    main()
