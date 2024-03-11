import time
import torch
import random
import argparse
import numpy as np
import seaborn as sns
import torch.nn.functional as F

from models import *
from tqdm import tqdm
from util_sparse import *
from dataset_loader import DataLoader
from utils import random_splits, random_splits_citation, fixed_splits, set_seed


def RunExp(args, dataset, data, Net, percls_trn, val_lb):
    def train(model, optimizer_Theta, optimizer_Q, data, W_cost, epoch):
        model.train()
        out_model = model(data, epoch)
        out = out_model[0][data.train_mask]
        # dual variable Z
        Z = torch.zeros_like(out)
        data_y = data.y[data.train_mask]
        data_y_one_hot = F.one_hot(data_y, num_classes=out.size(1))

        if args.Original_ot == 'ot':
            for i in range(args.iter_num_theta):
                #updata theta
                out, Q_update = model(data, epoch)[0][data.train_mask], model(data, epoch)[1].detach(),
                optimizer_Theta.zero_grad()
                loss_Theta = (torch.sum(Z * out)+
                              args.lambda_ * F.cross_entropy(out + J_all[data.train_mask] @ Q_update, data.y[data.train_mask]))
                loss_Theta.backward()
                optimizer_Theta.step()
                #print(f'loss_Theta:{loss_Theta}')
            out_update = model(data, epoch)[0][data.train_mask].detach()
            for i in range(args.iter_num_q):
                #updata Q
                Q = model(data, epoch)[1]
                optimizer_Q.zero_grad()
                loss_Q = (torch.sum(torch.linalg.norm(W_cost * Q, dim=0, ord=1)) +
                          torch.sum(Z * ( J_all[data.train_mask] @ Q)) +
                          args.lambda_ * F.cross_entropy(out_update + J_all[data.train_mask] @ Q, data.y[data.train_mask]))
                loss_Q.backward()
                optimizer_Q.step()
                #print(f'loss_Q:{loss_Q}')
            Q_update = model(data, epoch)[1].detach()
            #updata Z
            Z = Z + args.lambda_ *  (out_update + J_all[data.train_mask] @ Q_update - data_y_one_hot)
            Q = Q_update
        #del out
        return Q

    def test(model, data, Q, J_all, W_cost, epoch):
        model.eval()
        logits, accs, losses, preds = model(data, epoch)[0], [], [], []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            if args.Original_ot == 'ot':
                pred = torch.softmax(logits[mask] + J_all[mask, :] @ Q, dim=1).max(1)[1]
            elif args.Original_ot == 'Original':
                pred = torch.softmax(logits[mask], dim=1).max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            y_true = data.y[mask]
            if args.Original_ot == 'ot':
                loss = torch.sum(torch.linalg.norm(W_cost * Q, dim=0, ord=1)) + args.lambda_ * F.cross_entropy(
                    logits[mask] + J_all[mask, :] @ Q, y_true)
            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())
        return accs, preds, losses


    device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')
    if not args.full and args.dataset in ['Chameleon', 'Squirrel']:
        Net = ChebNetII_V
    tmp_net = Net(dataset, args)
    # Using the dataset splits described in the paper.
    if args.full:
        data = random_splits(data, dataset.num_classes, percls_trn, val_lb, args.seed)
    elif args.semi_rnd:
        if args.dataset in ["Cora", "Citeseer", "Pubmed"]:
            data = random_splits_citation(data, dataset.num_classes)
        else:
            data = random_splits(data, dataset.num_classes, percls_trn, val_lb, args.seed)
    elif args.semi_fix and args.dataset in ["Chameleon", "Squirrel", "Actor", "Texas", "Cornell"]:
        data = fixed_splits(data, dataset.num_classes, percls_trn, val_lb, args.dataset)

    model, data = tmp_net.to(device), data
    # J_all and W_cost for OT-GNN
    J_all, W_cost = J_W_cost(args, data, device)
    if args.Original_ot == 'ot':
        if args.net == 'GCN':
            optimizer_Theta = torch.optim.Adam(
                [{'params': model.conv1.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
                 {'params': model.conv2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr}
                 ])
            if args.A_F:
                optimizer_Q = torch.optim.Adam(
                    [{'params': model.Q, 'weight_decay': args.q_linear_delay, 'lr': args.q_linear_lr},
                     {'params': model.lin1.parameters(), 'weight_decay': args.q_linear_delay, 'lr': args.q_linear_lr},
                     {'params': model.lin2.parameters(), 'weight_decay': args.q_linear_delay, 'lr': args.q_linear_lr}
                     ])
            else:
                optimizer_Q = torch.optim.Adam(
                    [{'params': model.Q, 'weight_decay': args.q_linear_delay, 'lr': args.q_linear_lr}
                     ])
        elif args.net=='GAT':
            optimizer_Theta = torch.optim.Adam(
                [{'params': model.conv1.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
                 {'params': model.conv2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr}
                 ])
            optimizer_Q = torch.optim.Adam(
                [{'params': model.Q, 'weight_decay': args.q_linear_delay, 'lr': args.q_linear_lr}
                 ])
        elif args.net == 'GIN':
            optimizer_Theta = torch.optim.Adam(
                [{'params': model.conv1.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
                 {'params': model.conv2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr}
                 ])
            optimizer_Q = torch.optim.Adam(
                [{'params': model.Q, 'weight_decay': args.q_linear_delay, 'lr': args.q_linear_lr}
                 ])
        elif args.net=='GSAGE':
            optimizer_Theta = torch.optim.Adam(
                [{'params': model.conv1.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
                 {'params': model.conv2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr}
                 ])
            optimizer_Q = torch.optim.Adam(
                [{'params': model.Q, 'weight_decay': args.q_linear_delay, 'lr': args.q_linear_lr}
                 ])
        elif args.net=='APPNP':
            optimizer_Theta = torch.optim.Adam(
                [{'params': model.lin1.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
                 {'params': model.lin2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr}
                 ])
            optimizer_Q = torch.optim.Adam(
                [{'params': model.prop1.Q, 'weight_decay': args.q_linear_delay, 'lr': args.q_linear_lr}
                 ])

        elif args.net =='BernNet':
            optimizer_Theta = torch.optim.Adam(
                [{'params': model.lin1.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
                 {'params': model.lin2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
                 {'params': model.prop1.temp, 'weight_decay': 0.0, 'lr': args.Bern_lr}
                 ])
            if args.A_F:
                optimizer_Q = torch.optim.Adam(
                    [
                     {'params': model.prop1.Q, 'weight_decay': args.q_linear_delay, 'lr': args.q_linear_lr},
                     {'params': model.prop1.lin1.parameters(), 'weight_decay': args.q_linear_delay, 'lr': args.q_linear_lr},
                     {'params': model.prop1.lin2.parameters(), 'weight_decay': args.q_linear_delay, 'lr': args.q_linear_lr}
                     ])
            else:
                optimizer_Q = torch.optim.Adam(
                    [
                        {'params': model.prop1.Q, 'weight_decay': args.q_linear_delay, 'lr': args.q_linear_lr}
                    ])
        
        elif args.net in ['ChebNetII']:
            optimizer_Theta = torch.optim.Adam(
                [{'params': model.lin1.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
                 {'params': model.lin2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
                 {'params': model.prop1.temp, 'weight_decay': args.prop_wd, 'lr': args.prop_lr}
                 ])
            if args.A_F:
                optimizer_Q = torch.optim.Adam(
                    [ {'params': model.prop1.Q, 'weight_decay': args.q_linear_delay, 'lr': args.q_linear_lr},
                      {'params': model.prop1.lin1.parameters(), 'weight_decay': args.q_linear_delay,
                       'lr': args.q_linear_lr},
                      {'params': model.prop1.lin2.parameters(), 'weight_decay': args.q_linear_delay,
                       'lr': args.q_linear_lr}
                    ])
            else:
                optimizer_Q = torch.optim.Adam(
                    [{'params': model.prop1.Q, 'weight_decay': args.q_linear_delay, 'lr': args.q_linear_lr}
                     ])

    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []

    time_run = []
    for epoch in range(args.epochs):
        t_st = time.time()
        Q = train(model, optimizer_Theta, optimizer_Q, data, W_cost, epoch)
        time_epoch = time.time() - t_st  # each epoch train times
        time_run.append(time_epoch)
        [train_acc, val_acc, tmp_test_acc], preds, [train_loss, val_loss, tmp_test_loss] = test(model, data, Q, J_all, W_cost, epoch)

        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            if args.net in ['ChebNetII']:
                TEST = tmp_net.prop1.temp.clone()
                theta = TEST.detach().cpu()
                theta = torch.relu(theta).numpy()
            elif args.net in ['ChebBase']:
                TEST = tmp_net.prop1.temp.clone()
                theta = TEST.detach().cpu().numpy()
            else:
                theta = args.alpha

        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    break
    return test_acc, best_val_acc, theta, time_run


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='seed.')
    parser.add_argument('--epochs', type=int, default=1000, help='max epochs.')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay.')
    parser.add_argument('--early_stopping', type=int, default=200, help='early stopping.')
    parser.add_argument('--hidden', type=int, default=64, help='hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout for neural networks.')

    parser.add_argument('--train_rate', type=float, default=0.6, help='train set rate.')
    parser.add_argument('--val_rate', type=float, default=0.2, help='val set rate.')
    parser.add_argument('--K', type=int, default=10, help='propagation steps.')
    
    parser.add_argument('--dprate', type=float, default=0.5, help='dropout for propagation layer.')

    parser.add_argument('--dataset', type=str,
                        choices=['Cora', 'Photo', 'Citeseer', 'Pubmed', 'Computers', 'Chameleon', 'Squirrel', 'Actor', 'Texas', 'Cornell'],
                        default='Texas')
    parser.add_argument('--device', type=int, default=0, help='GPU device.')
    parser.add_argument('--runs', type=int, default=10, help='number of runs.')
    parser.add_argument('--net', type=str, choices=['GCN', 'GAT', "GIN", "GSAGE", 'APPNP', 'BernNet', 'ChebNetII'],
                        default='BernNet')
     #ChenNetII
    parser.add_argument('--prop_lr', type=float, default=0.01, help='learning rate for propagation layer.')
    parser.add_argument('--prop_wd', type=float, default=0.0, help='learning rate for propagation layer.')

    #BernNet
    parser.add_argument('--Bern_lr', type=float, default=0.01, help='learning rate for BernNet propagation layer.')

    #GAT
    parser.add_argument('--heads', default=8, type=int, help='attention heads for GAT.')
    parser.add_argument('--output_heads', default=1, type=int, help='output_heads for GAT.')

    #APPNP
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha for APPN.')

    parser.add_argument('--q', type=int, default=0, help='The constant for ChebBase.')
    parser.add_argument('--full', type=bool, default=True, help='full-supervise with random splits')
    parser.add_argument('--semi_rnd', type=bool, default=False, help='semi-supervised with random splits')
    parser.add_argument('--semi_fix', type=bool, default=False, help='semi-supervised with fixed splits')

    parser.add_argument('--lambda_', type=float, default=0.001, help='')
    parser.add_argument('--A_F', action='store_true', help='')
    parser.add_argument('--mlp_layers', type=int, default=2, help='')
    parser.add_argument('--q_linear_lr', type=float, default=0.05, help='q_linear_lr.')
    parser.add_argument('--q_linear_delay', type=float, default=0.0005, help='q_linear_delay.')
    parser.add_argument('--Original_ot', type=str, choices=['ot'], default='ot')
    parser.add_argument('--com_w_cost', type=str, choices=['ones'], default='ones')
    parser.add_argument('--activation', type=str, choices=['sigmoid'], default='sigmoid')
    parser.add_argument('--iter_num_theta', type=int, default=1, help='')
    parser.add_argument('--iter_num_q', type=int, default=1, help='')

    args = parser.parse_args()
    set_seed(args.seed)
    # 10 fixed seeds for random splits from BernNet
    SEEDS = [1941488137, 4198936517, 983997847, 4023022221, 4019585660, 2108550661, 1648766618, 629014539, 3212139042,
             2424918363]

    print(args)
    print("---------------------------------------------")

    gnn_name = args.net
    if gnn_name == 'GCN':
        Net = GCN_Net
    elif gnn_name == 'GAT':
        Net = GAT_Net
    elif gnn_name == 'GIN':
        Net = GIN_Net
    elif gnn_name == 'GSAGE':
        Net = GSAGE_Net
    elif gnn_name == 'APPNP':
        Net = APPNP_Net
    elif gnn_name == 'BernNet':
        Net = BernNet
    elif gnn_name == "ChebNetII":
        Net = ChebNetII

    device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')
    dataset = DataLoader(args.dataset)
    data = dataset[0].to(device)
    mask = data.edge_index[0, :] < data.edge_index[1, :]
    edge_index = data.edge_index[:, mask]
    edge_index2 = torch.cat([edge_index[1, :].unsqueeze(0), edge_index[0, :].unsqueeze(0)], dim=0)
    data.edge_index = torch.cat([edge_index, edge_index2], dim=1)
    dataset.data.edge_index = data.edge_index



    if args.full:
        args.train_rate = args.train_rate
        args.val_rate = args.val_rate
    else:
        args.train_rate = 0.025
        args.val_rate = 0.025
    percls_trn = int(round(args.train_rate * len(data.y) / dataset.num_classes))
    val_lb = int(round(args.val_rate * len(data.y)))

    results = []
    time_results = []
    for RP in tqdm(range(args.runs)):
        args.seed = SEEDS[RP]
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        test_acc, best_val_acc, theta_0, time_run = RunExp(args, dataset, data, Net, percls_trn, val_lb)
        time_results.append(time_run)
        results.append([test_acc, best_val_acc])
        print(f'run_{str(RP + 1)} \t test_acc: {test_acc:.4f}')
        if args.net in ["ChebBase", "ChebNetII"]:
            print('Weights:', [float('{:.4f}'.format(i)) for i in theta_0])

    run_sum = 0
    epochsss = 0
    for i in time_results:
        run_sum += sum(i)
        epochsss += len(i)
    print("each run avg_time:", run_sum / (args.runs), "s")
    print("each epoch avg_time:", 1000 * run_sum / epochsss, "ms")
    test_acc_mean, val_acc_mean = np.mean(results, axis=0) * 100
    test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100
    values = np.asarray(results, dtype=object)[:, 0]
    uncertainty = np.max(
        np.abs(sns.utils.ci(sns.algorithms.bootstrap(values, func=np.mean, n_boot=1000), 95) - values.mean()))
    print(f'{gnn_name} on dataset {args.dataset}, in {args.runs} repeated experiment:')
    print(f'test acc mean = {test_acc_mean:.4f} ± {uncertainty * 100:.4f}  \t val acc mean = {val_acc_mean:.4f}')

    with open('logs-2-' + args.net + '_QW_' + args.dataset, 'a+') as f:
        f.write(
            'net:{}, Original_ot:{}, train_rate:{}, val_rate:{}, lr:{}, weight_decay:{}, dropout:{}, prop_lr:{}, prop_wd:{}, dprate:{}, lambda_:{}, q_linear_lr:{}, q_linear_delay:{}, dataset:{}, iter_num_theta:{}, iter_num_q:{}, A_F:{} \n'.format(
                args.net, args.Original_ot, args.train_rate, args.val_rate,
                args.lr, args.weight_decay, args.dropout,
                args.prop_lr, args.prop_wd, args.dprate,
                args.lambda_, args.q_linear_lr, args.q_linear_delay, args.dataset, args.iter_num_theta, args.iter_num_q, args.A_F
                ))
        f.write(f'test acc mean = {test_acc_mean:.2f} ± {uncertainty * 100:.2f}  \t val acc mean = {val_acc_mean:.4f}')
        f.write('\n')
        f.write('\n')
