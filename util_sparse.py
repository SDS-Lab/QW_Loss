import torch
import numpy as np
import matplotlib.pyplot as plt

def J_W_cost(args, data, device):
    """
    Calculate J_all and W_cost for OT-GNN
    :param dataset
    :return:
        J_all: (N, E), N and E represent the number of nodes and edges in a graph
        W_cost: (E, E)
    """
    J_all = torch.zeros(int(data.x.size(0)), int(data.edge_index.size(1) / 2)).to(device)
    w_diag = []
    for i in range(int(data.edge_index.size(1) / 2)):
        row_1, row_2 = data.edge_index[:, i]
        J_all[row_1][i] = 1
        J_all[row_2][i] = -1
        if args.com_w_cost == 'cosine':
            w_diag.append(torch.nn.functional.cosine_similarity(data.x[row_1, :], data.x[row_2, :], dim=0))
        elif args.com_w_cost == 'ones':
            w_diag.append(1.0)
        elif args.com_w_cost == '1_1/2cosine':
            w_diag.append(1 - 0.5 * torch.nn.functional.cosine_similarity(data.x[row_1, :], data.x[row_2, :],dim=0))
        elif args.com_w_cost == 'inner_product':
            w_diag.append(torch.sum(data.x[row_1, :] * data.x[row_2, :]))

    #W_cost = torch.diag(torch.abs(torch.tensor(w_diag)))
    #W_cost = W_cost.to_sparse().to(device)
    #J_all = J_all.to_sparse().to(device)
    return J_all.to(device), torch.tensor(w_diag).unsqueeze(1).to(device)


def edge_weight_plot(Q, W, data, dataset):
    """
    Draw a heatmap of the weight matrix of edges.
    :param Q: (E, C), E and C represent the number of edges and the classes of nodes
    :param W: (C, 1)
    :return:
        data.y[int(x)]: The label of node x
    """

    def node_label(x):
        """
        Obtain the labels corresponding to the nodes, in order to calculate the map of edge weights.
        :param x
        :return:
            data.y[int(x)]: The label of node x
        """
        return data.y[int(x)]

    edge_weight = Q @ W
    edge_weight = edge_weight.detach().cpu()
    edge_index = data.edge_index[:, :int(data.edge_index.size(1)/2)].detach().cpu()
    edge_node = edge_index.apply_(node_label)

    weight_heatmap = np.zeros((dataset.num_classes,dataset.num_classes))
    num_heatmap = np.zeros((dataset.num_classes,dataset.num_classes))
    for i in range(edge_index.size(1)):
        source_index, target_index = edge_node[:,i]
        min_index, max_index = np.minimum(source_index, target_index),np.maximum(source_index, target_index)
        #print(edge_weight[i])
        weight_heatmap[min_index,max_index] = weight_heatmap[min_index,max_index] + edge_weight[i]
        num_heatmap[min_index,max_index] = num_heatmap[min_index,max_index] + 1
    num_heatmap = np.where(num_heatmap==0,1,num_heatmap)
    weight_heatmap = weight_heatmap/num_heatmap
    weight_map = weight_heatmap + weight_heatmap.T - np.diag(weight_heatmap.diagonal())
    plt.imshow(weight_map, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.show()