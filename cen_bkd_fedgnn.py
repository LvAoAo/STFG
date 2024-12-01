from Common.Node.workerbasev2 import WorkerBaseV2
import torch
from torch import nn
from torch import device
import json
import os

from Common.Utils.options import args_parser
from Common.Utils.gnn_util import inject_global_trigger_test, inject_global_trigger_train, load_pkl, split_dataset
import time
from Common.Utils.evaluate import gnn_evaluate_accuracy_v2
import numpy as np 
import torch.nn.functional as F
from GNN_common.data.TUs import TUsDataset
from GNN_common.nets.TUs_graph_classification.load_net import gnn_model  # import GNNs
from torch.utils.data import DataLoader
from defense import foolsgold
import copy
import wandb
import tqdm

def server_robust_agg(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg
    
class ClearDenseClient(WorkerBaseV2):
    def __init__(self, client_id, model, loss_func, train_iter, attack_iter, test_iter, config, optimizer, device, grad_stub, args, scheduler):
        super(ClearDenseClient, self).__init__(model=model, loss_func=loss_func, train_iter=train_iter, attack_iter=attack_iter,
                                               test_iter=test_iter, config=config, optimizer=optimizer, device=device)
        self.client_id = client_id
        self.grad_stub = None
        self.args = args
        self.scheduler = scheduler

    def update(self):
        pass

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


if __name__ == '__main__':
    args = args_parser()
    wandb.init(
        project="fed_backdoor",
        group=f"{args.dataset}_centralized",
        config=args
    )
    torch.manual_seed(args.seed)
    with open(args.config) as f:
        config = json.load(f)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset = TUsDataset(args)

    collate = dataset.collate
    MODEL_NAME = config['model']
    net_params = config['net_params']
    if MODEL_NAME in ['GCN', 'GAT']:
        if net_params['self_loop']:
            print("[!] Adding graph self-loops for GCN/GAT models (central node trick).")
            dataset._add_self_loops()
    net_params['in_dim'] = dataset.all.graph_lists[0].ndata['feat'][0].shape[0]
    num_classes = torch.max(dataset.all.graph_labels).item() + 1
    net_params['n_classes'] = num_classes
    net_params['dropout'] = args.dropout

    ## set a global model
    global_model = gnn_model(MODEL_NAME, net_params)
    global_model = global_model.to(device)
    #print("Target Model:\n{}".format(model))
    client = []
    loss_func = nn.CrossEntropyLoss()
    # Load data
    partition, avg_nodes = split_dataset(args, dataset)
    drop_last = True if MODEL_NAME == 'DiffPool' else False
    filename = "./Data/global_trigger/%d/%s_%s_%d_%d_%d_%.2f_%.2f_%.2f"\
              %(args.seed, MODEL_NAME, config['dataset'], args.num_workers, args.num_mali, args.epoch_backdoor, args.frac_of_avg, args.poisoning_intensity, args.density) + '.pkl'
    global_trigger = load_pkl(filename)
    print("Triggers loaded!")
    args.num_mali = len(global_trigger)
    for i in range(args.num_workers):
        local_model = copy.deepcopy(global_model)
        local_model = local_model.to(device)
        optimizer = torch.optim.Adam(local_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=args.step_size, gamma=args.gamma)

        print("Client %d training data num: %d"%(i, len(partition[i])))
        print("Client %d testing data num: %d"%(i, len(partition[-1])))
        train_loader = DataLoader(partition[i], batch_size=args.batch_size, shuffle=True,
                                    drop_last=drop_last,
                                    collate_fn=dataset.collate)
        attack_loader = None
        test_loader = DataLoader(partition[-1], batch_size=args.batch_size, shuffle=True,
                                    drop_last=drop_last,
                                    collate_fn=dataset.collate)
        
        client.append(ClearDenseClient(client_id=i, model=local_model, loss_func=loss_func, train_iter=train_loader, attack_iter=attack_loader, test_iter=test_loader, config=config, optimizer=optimizer, device=device, grad_stub=None, args=args, scheduler=scheduler))
    # check model memory address
    for i in range(args.num_workers):
        add_m = id(client[i].model)
        add_o = id(client[i].optimizer)
        add_s = id(client[i].scheduler)
        print('model {} address: {}'.format(i, add_m))
        print('optimizer {} address: {}'.format(i, add_o))
        print('scheduler {} address: {}'.format(i, add_s))
    # prepare backdoor training dataset and testing dataset
    train_trigger_graphs, final_idx = inject_global_trigger_train(partition[0], avg_nodes, args, global_trigger)
    test_trigger_graphs = inject_global_trigger_test(partition[-1], avg_nodes, args, global_trigger)
    tmp_graphs = [partition[0][idx] for idx in range(len(partition[0])) if idx not in final_idx]

    train_dataset = train_trigger_graphs + tmp_graphs
    backdoor_train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                            drop_last=drop_last,
                            collate_fn=dataset.collate)
    backdoor_attack_loader = DataLoader(test_trigger_graphs, batch_size=args.batch_size, shuffle=True,
                            drop_last=drop_last,
                            collate_fn=dataset.collate)
    test_local_trigger_load = []
    for i in range(len(global_trigger)):
        test_local_trigger = inject_global_trigger_test(partition[-1], avg_nodes, args, [global_trigger[i]])
        tmp_load = DataLoader(test_local_trigger, batch_size=args.batch_size, shuffle=True,
                            drop_last=drop_last,
                            collate_fn=dataset.collate)
        test_local_trigger_load.append(tmp_load)
    acc_record = [0]
    counts = 0
    weight_history = []
    for epoch in tqdm.tqdm(range(args.epochs)):
        if epoch >= args.epoch_backdoor:
            # inject global trigger into the centrilized attacker - client[0]
            client[0].train_iter = backdoor_train_loader
            client[0].attack_iter = backdoor_attack_loader

        for i in range(args.num_workers):
            att_list = []
            train_loss, train_acc, test_loss, test_acc = client[i].gnn_train_v2()
            client[i].scheduler.step()
            global_att = gnn_evaluate_accuracy_v2(backdoor_attack_loader, client[i].model)
            
            # Log client metrics
            metrics = {
                f'client_{i}/train_loss': train_loss,
                f'client_{i}/train_acc': train_acc,
                f'client_{i}/test_loss': test_loss,
                f'client_{i}/test_acc': test_acc,
                f'client_{i}/global_trigger_acc': global_att
            }
            
            # Log local trigger accuracies
            for j in range(len(global_trigger)):
                tmp_acc = gnn_evaluate_accuracy_v2(test_local_trigger_load[j], client[i].model)
                metrics[f'client_{i}/local_trigger_{j}_acc'] = tmp_acc
                att_list.append(tmp_acc)
            
            wandb.log(metrics, step=epoch)

        weights = []
        for i in range(args.num_workers):
            weights.append(client[i].get_weights())
            weight_history.append(client[i].get_weights_list())
        # Aggregation in the server to get the global model
        if args.defense == 'foolsgold':
            result, weight_history, alpha = foolsgold(args, weight_history, weights, global_model, client[0])
            wandb.log({f'defense/alpha_{i}': alpha[i] for i in range(args.num_workers)}, step=epoch)
        else:
            result = server_robust_agg(weights)

        for i in range(args.num_workers):
            client[i].set_weights(weights=result)
            client[i].upgrade()
        # update global model's weights
        global_model.load_state_dict(result)
        
        # evaluate the global model: test_acc
        test_acc = gnn_evaluate_accuracy_v2(client[0].test_iter, global_model)
        metrics = {
            'global/test_acc': test_acc
        }

        if epoch >= args.epoch_backdoor:
            global_att_acc = gnn_evaluate_accuracy_v2(backdoor_attack_loader, global_model)
            metrics['global/global_trigger_acc'] = global_att_acc
            
            for i in range(len(global_trigger)):
                local_trigger_acc = gnn_evaluate_accuracy_v2(test_local_trigger_load[i], global_model)
                metrics[f'global/local_trigger_{i}_acc'] = local_trigger_acc
            
        wandb.log(metrics, step=epoch)
        
                

