from Common.Node.workerbasev2 import WorkerBaseV2, ClearDenseClient, ClearSparseClient
import torch
from torch import nn
from torch import device
import json
import os
from Common.Utils.options import args_parser
from Common.Utils.gnn_util import transform_dataset, inject_global_trigger_test, save_object, split_dataset
from Common.Utils.lockdown_util import calculate_sparsities, init_masks
import time
from Common.Utils.evaluate import gnn_evaluate_accuracy_v2
import numpy as np 
import torch.nn.functional as F
from GNN_common.data.TUs import TUsDataset
from GNN_common.nets.TUs_graph_classification.load_net import gnn_model  # import GNNs
from torch.utils.data import DataLoader
from defense import foolsgold, Robust_Learning_Rate, Aggregation
import copy
from torch.nn.utils import vector_to_parameters, parameters_to_vector
import wandb
import tqdm


def server_robust_agg(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg
    


class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


if __name__ == '__main__':
    args = args_parser()
    wandb.init(
        project="fed_backdoor",
        group=f"{args.dataset}_distributed",
        config=args
    )
    torch.manual_seed(args.seed)
    with open(args.config) as f:
        config = json.load(f)
        wandb.config.update(config)
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

    global_model = gnn_model(MODEL_NAME, net_params)
    global_model = global_model.to(device)
    # global_mask = {}
    n_model_params = len(parameters_to_vector([ global_model.state_dict()[name] for name in global_model.state_dict()]))
    params = {name: copy.deepcopy(global_model.state_dict()[name]) for name in global_model.state_dict()}
    if args.defense == "lockdown":
        sparsity = calculate_sparsities(args, params, distribution=args.mask_init)
        mask = init_masks(params, sparsity)

    #print("Target Model:\n{}".format(model))
    client = []
    loss_func = nn.CrossEntropyLoss()
    # Load data
    partition, avg_nodes = split_dataset(args, dataset)
    drop_last = True if MODEL_NAME == 'DiffPool' else False
    triggers = []
    agent_data_sizes = {}
    for i in range(args.num_workers):
        local_model = copy.deepcopy(global_model)
        local_model = local_model.to(device)
        optimizer = torch.optim.Adam(local_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=args.step_size, gamma=args.gamma)

        train_dataset = partition[i]
        test_dataset = partition[-1]
        print("Client %d training data num: %d"%(i, len(train_dataset)))
        print("Client %d testing data num: %d"%(i, len(test_dataset)))
        agent_data_sizes[i] = len(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                     drop_last=drop_last,
                                     collate_fn=dataset.collate)
        attack_loader = None
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                                     drop_last=drop_last,
                                     collate_fn=dataset.collate)
        
        if args.defense == "lockdown":
            if args.same_mask==0:
                agent = ClearSparseClient(client_id=i, model=local_model, loss_func=loss_func, train_iter=train_loader, attack_iter=attack_loader, test_iter=test_loader, config=config, optimizer=optimizer, device=device, grad_stub=None, args=args, scheduler=scheduler, mask=init_masks(params, sparsity))
            else:
                agent = ClearSparseClient(client_id=i, model=local_model, loss_func=loss_func, train_iter=train_loader, attack_iter=attack_loader, test_iter=test_loader, config=config, optimizer=optimizer, device=device, grad_stub=None, args=args, scheduler=scheduler, mask=mask)
        else:
            agent = ClearDenseClient(client_id=i, model=local_model, loss_func=loss_func, train_iter=train_loader, attack_iter=attack_loader, test_iter=test_loader, config=config, optimizer=optimizer, device=device, grad_stub=None, args=args, scheduler=scheduler)
        client.append(agent)
    # check model memory address
    for i in range(args.num_workers):
        add_m = id(client[i].model)
        add_o = id(client[i].optimizer)
        print('model {} address: {}'.format(i, add_m))
        print('optimizer {} address: {}'.format(i, add_o))
    # prepare backdoor local backdoor dataset
    train_loader_list = []
    attack_loader_list = []
    for i in range(args.num_mali):
        train_trigger_graphs, test_trigger_graphs, G_trigger, final_idx = transform_dataset(partition[i], partition[-1], avg_nodes, args)
        triggers.append(G_trigger)
        tmp_graphs = [partition[i][idx] for idx in range(len(partition[i])) if idx not in final_idx]
        # backdoored graphs + clean graphs
        train_dataset = train_trigger_graphs + tmp_graphs
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                drop_last=drop_last,
                                collate_fn=dataset.collate)
        attack_loader = DataLoader(test_trigger_graphs, batch_size=args.batch_size, shuffle=True,
                                drop_last=drop_last,
                                collate_fn=dataset.collate)
        train_loader_list.append(train_loader)
        attack_loader_list.append(attack_loader)
    # save global trigger in order to implement centrilized backoor attack
    if args.num_mali > 0:
        filename = "./Data/global_trigger/%d/%s_%s_%d_%d_%d_%.2f_%.2f_%.2f"\
            %(args.seed, MODEL_NAME, config['dataset'], args.num_workers, args.num_mali, args.epoch_backdoor, args.frac_of_avg, args.poisoning_intensity, args.density) + '.pkl'
        path = os.path.split(filename)[0]
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        # save the global trigger that is then used in the centralized backdoor attack
        save_object(triggers, filename)
        print('The global trigger is saved successfully!')
        test_global_trigger = inject_global_trigger_test(partition[-1], avg_nodes, args, triggers)
        test_global_trigger_load = DataLoader(test_global_trigger, batch_size=args.batch_size, shuffle=True,
                                drop_last=drop_last,
                                collate_fn=dataset.collate)
    acc_record = [0]
    counts = 0
    weight_history = []
    for epoch in tqdm.tqdm(range(args.epochs)):
        agent_updates_dict = {}
        
        if epoch >= args.epoch_backdoor:
            for i in range(0, args.num_mali):
                client[i].train_iter = train_loader_list[i]
                client[i].attack_iter = attack_loader_list[i]
        if args.defense == "lockdown":
            old_mask = [copy.deepcopy(client[i].mask) for i in range(args.num_workers)]
        for i in range(args.num_workers):
            att_list = []
            if args.defense == "lockdown":
                train_loss, train_acc, test_loss, test_acc = client[i].gnn_train_v2(round=epoch)
            else:
                train_loss, train_acc, test_loss, test_acc = client[i].gnn_train_v2()
            update = client[i].get_update()
            agent_updates_dict[i] = update
            # print(f"Client {i} update: {update}")
            
            global_att = gnn_evaluate_accuracy_v2(test_global_trigger_load, client[i].model)
            
            # Log client metrics
            metrics = {
                f'client_{i}/train_loss': train_loss,
                f'client_{i}/train_acc': train_acc,
                f'client_{i}/test_loss': test_loss,
                f'client_{i}/test_acc': test_acc,
                f'client_{i}/global_trigger_acc': global_att
            }
            # if i == 0:
            #     print(f"Client {i} metrics: {metrics}")
            
            # Log local trigger accuracies
            for j in range(len(triggers)):
                tmp_acc = gnn_evaluate_accuracy_v2(attack_loader_list[j], client[i].model)
                metrics[f'client_{i}/local_trigger_{j}_acc'] = tmp_acc
                att_list.append(tmp_acc)
            
            wandb.log(metrics, step=epoch)

        weights = []
        for i in range(args.num_workers):
            weights.append(client[i].get_weights())
            weight_history.append(client[i].get_weights_list())
        # Aggregation in the server to get the global model
        # if there is a defense applied
        if args.defense == 'foolsgold':
            result, weight_history, alpha = foolsgold(args, weight_history, weights, global_model, client[0])
            wandb.log({f'defense/alpha_{i}': alpha[i] for i in range(args.num_workers)}, step=epoch)
        elif args.defense == 'rlr':
            n_params = len(parameters_to_vector(global_model.parameters()))
            aggregator = Robust_Learning_Rate(agent_data_sizes, n_params, args)
            result = aggregator.aggregate_updates(global_model, agent_updates_dict)
            # result = global_model.state_dict()
        elif args.defense == "lockdown":
            # print("Lockdown defense")
            # n_params = len(parameters_to_vector([ global_model.state_dict()[name] for name in global_model.state_dict()]))
            # aggregator = Aggregation(agent_data_sizes, n_params, args)
            # result, _ = aggregator.aggregate_updates(global_model, agent_updates_dict)
            # TODO: Add aggregation function to lockdown and rlr
            result = server_robust_agg(weights)
        else:
            # average aggregation
            result = server_robust_agg(weights)
        # print(f"Global model update: {result}")
        for i in range(args.num_workers):
            client[i].set_weights(weights=result)
            client[i].upgrade()
        # update global model's weights
        # Add before and after update checks
        # old_params = parameters_to_vector(global_model.parameters()).clone()
        global_model.load_state_dict(result)
        # new_params = parameters_to_vector(global_model.parameters()).clone()
        # param_diff = torch.norm(new_params - old_params)
        # print(f"Parameter update magnitude: {param_diff}")
        metrics = {}
        if args.defense == "lockdown":
            test_model = copy.deepcopy(global_model)
            for name, param in test_model.named_parameters():
                mask = 0
                for id, agent in enumerate(client):
                    mask += old_mask[id][name].to(args.device)
                param.data = torch.where(mask.to(args.device) >= args.theta, param,
                                        torch.zeros_like(param))
        else:
            test_model = copy.deepcopy(global_model)
        test_model.eval()
        test_acc = gnn_evaluate_accuracy_v2(client[0].test_iter, test_model)
        metrics['global/test_acc'] = test_acc
        # print(test_acc)
        

        # Log global model trigger accuracies
        if args.num_mali > 0 and epoch >= args.epoch_backdoor:
            global_att_acc = gnn_evaluate_accuracy_v2(test_global_trigger_load, test_model)
            metrics['global/global_trigger_acc'] = global_att_acc
            
            for i in range(args.num_mali):
                local_trigger_acc = gnn_evaluate_accuracy_v2(attack_loader_list[i], test_model)
                metrics[f'global/local_trigger_{i}_acc'] = local_trigger_acc

        wandb.log(metrics, step=epoch)
        del test_model
