import hdbscan
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from scipy.special import logit
import sklearn.metrics.pairwise as smp
import ipdb
import torch
import copy
from scipy.special import softmax
from torch.nn.utils import vector_to_parameters, parameters_to_vector
from Common.Utils.lockdown_util import vector_to_model, vector_to_name_param
import logging

def fedavg(args, grad_in):
    grad = np.array(grad_in).reshape((args.num_workers, -1)).mean(axis=0)
    return grad.tolist()

def foolsgold(args, grad_history, grad_in, global_model, client):
    epsilon = 1e-5
    grad_history = np.array(grad_history)
    if grad_history.shape[0] != args.num_workers:
        grad_history = grad_history[:args.num_workers,:] + grad_history[args.num_workers:,:]

    similarity_maxtrix = smp.cosine_similarity(grad_history) - np.eye(args.num_workers)

    mv = np.max(similarity_maxtrix, axis=1) + epsilon

    alpha = np.zeros(mv.shape)
    for i in range(args.num_workers):
        for j in range(args.num_workers):
            if mv[j] > mv[i]:
                similarity_maxtrix[i,j] *= mv[i]/mv[j]

    alpha = 1 - (np.max(similarity_maxtrix, axis=1))
    alpha[alpha > 1] = 1
    alpha[alpha < 0] = 0
    alpha = alpha/np.max(alpha)
    alpha[(alpha == 1)] = 0.99
    alpha = (np.log((alpha / (1 - alpha)) + epsilon) + 0.5)
    alpha[(np.isinf(alpha) + alpha > 1)] = 1
    alpha[(alpha < 0)] = 0
    print("alpha:")
    print(alpha)

    # softmax alpha to make it summing up to 1
    alpha = softmax(alpha)
    update_weights = copy.deepcopy(grad_in[0])
    for key in grad_in[0].keys():
        update_weights[key] = update_weights[key] * alpha[0]
        for i in range(1, len(grad_in)):
            update_weights[key] += grad_in[i][key] * alpha[i]
        update_weights[key] = torch.div(update_weights[key], len(grad_in))
    
    ipdb.set_trace
    is_nan = torch.stack([torch.isnan(p).any() for p in global_model.parameters()]).any()
    if is_nan.item():
        print('The model has nan values')
        ipdb.set_trace()
    return update_weights, grad_history.tolist(), alpha


class Robust_Learning_Rate():
    def __init__(self, agent_data_sizes, n_params, args):
        self.agent_data_sizes = agent_data_sizes
        self.args = args
        self.n_params = n_params

    def aggregate_updates(self, global_model, agent_updates_dict):
        # adjust LR if robust LR is selected
        lr_vector = torch.Tensor([self.args.server_lr]*self.n_params).to(self.args.device)
        if self.args.robustLR_threshold > 0:
            lr_vector = self.compute_robustLR(agent_updates_dict)
        
        print(lr_vector)
        aggregated_updates = 0
        aggregated_updates = self.agg_comed(agent_updates_dict)      
                
        cur_global_params = parameters_to_vector(global_model.parameters())
        new_global_params =  (cur_global_params + lr_vector*aggregated_updates).float() 
        vector_to_parameters(new_global_params, global_model.parameters())
        updated_weights = global_model.state_dict()
        return updated_weights

    def compute_robustLR(self, agent_updates_dict):
        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]  
        sm_of_signs = torch.abs(sum(agent_updates_sign))
        
        sm_of_signs[sm_of_signs < self.args.robustLR_threshold] = -self.args.server_lr
        sm_of_signs[sm_of_signs >= self.args.robustLR_threshold] = self.args.server_lr                                            
        return sm_of_signs.to(self.args.device)
    
    def agg_avg(self, agent_updates_dict):
        sm_updates = 0
        for _id, update in agent_updates_dict.items():
            sm_updates += update
        return sm_updates / len(agent_updates_dict.keys())
    
    def agg_comed(self, agent_updates_dict):
        agent_updates_col_vector = [update.view(-1, 1) for update in agent_updates_dict.values()]
        concat_col_vectors = torch.cat(agent_updates_col_vector, dim=1)
        return torch.median(concat_col_vectors, dim=1).values
    

class Aggregation():
    def __init__(self, agent_data_sizes, n_params, args):
        self.agent_data_sizes = agent_data_sizes
        self.args = args
        self.server_lr = args.server_lr
        self.n_params = n_params
        self.cum_net_mov = 0
        
         
    def aggregate_updates(self, global_model, agent_updates_dict):
        lr_vector = torch.Tensor([self.server_lr]*self.n_params).to(self.args.device)
        if self.args.defense != "rlr":
            lr_vector=lr_vector
        else:
            lr_vector, _ = self.compute_robustLR(agent_updates_dict)
        # mask = torch.ones_like(agent_updates_dict[0])
        aggregated_updates = 0
        cur_global_params = parameters_to_vector(
            [global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        if self.args.aggr=='avg':          
            aggregated_updates = self.agg_avg(agent_updates_dict)
        if self.args.aggr== "clip_avg":
            for _id, update in agent_updates_dict.items():
                weight_diff_norm = torch.norm(update).item()
                logging.info(weight_diff_norm)
                update.data = update.data / max(1, weight_diff_norm / 2)
            aggregated_updates = self.agg_avg(agent_updates_dict)
            logging.info(torch.norm(aggregated_updates))
        elif self.args.aggr=='comed':
            aggregated_updates = self.agg_comed(agent_updates_dict)
        elif self.args.aggr == 'sign':
            aggregated_updates = self.agg_sign(agent_updates_dict)
        elif self.args.aggr == "krum":
            aggregated_updates = self.agg_krum(agent_updates_dict)
        elif self.args.aggr == "gm":
            aggregated_updates = self.agg_gm(agent_updates_dict,cur_global_params)
        elif self.args.aggr == "tm":
            aggregated_updates = self.agg_tm(agent_updates_dict)
        neurotoxin_mask = {}
        updates_dict = vector_to_name_param(aggregated_updates, copy.deepcopy(global_model.state_dict()))
        for name in updates_dict:
            updates = updates_dict[name].abs().view(-1)
            gradients_length = torch.numel(updates)
            _, indices = torch.topk(-1 * updates, int(gradients_length * self.args.dense_ratio))
            mask_flat = torch.zeros(gradients_length)
            mask_flat[indices.cpu()] = 1
            neurotoxin_mask[name] = (mask_flat.reshape(updates_dict[name].size()))

        cur_global_params = parameters_to_vector([ global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        new_global_params =  (cur_global_params + lr_vector*aggregated_updates).float()
        vector_to_model(new_global_params, global_model)
        updated_weights = global_model.state_dict()
        return updated_weights, neurotoxin_mask


    def compute_robustLR(self, agent_updates_dict):

        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]  
        sm_of_signs = torch.abs(sum(agent_updates_sign))
        mask=torch.zeros_like(sm_of_signs)
        mask[sm_of_signs < self.args.theta] = 0
        mask[sm_of_signs >= self.args.theta] = 1
        sm_of_signs[sm_of_signs < self.args.theta] = -self.server_lr
        sm_of_signs[sm_of_signs >= self.args.theta] = self.server_lr
        return sm_of_signs.to(self.args.device), mask


    def agg_krum(self, agent_updates_dict):
        krum_param_m = 1
        def _compute_krum_score( vec_grad_list, byzantine_client_num):
            krum_scores = []
            num_client = len(vec_grad_list)
            for i in range(0, num_client):
                dists = []
                for j in range(0, num_client):
                    if i != j:
                        dists.append(
                            torch.norm(vec_grad_list[i]- vec_grad_list[j])
                            .item() ** 2
                        )
                dists.sort()  # ascending
                score = dists[0: num_client - byzantine_client_num - 2]
                krum_scores.append(sum(score))
            return krum_scores

        # Compute list of scores
        __nbworkers = len(agent_updates_dict)
        krum_scores = _compute_krum_score(agent_updates_dict, self.args.num_corrupt)
        score_index = torch.argsort(
            torch.Tensor(krum_scores)
        ).tolist()  # indices; ascending
        score_index = score_index[0: krum_param_m]
        return_gradient = [agent_updates_dict[i] for i in score_index]
        return sum(return_gradient)/len(return_gradient)

    def agg_avg(self, agent_updates_dict):
        """ classic fed avg """

        sm_updates, total_data = 0, 0
        for _id, update in agent_updates_dict.items():
            n_agent_data = self.agent_data_sizes[_id]
            sm_updates +=  n_agent_data * update
            total_data += n_agent_data
        return  sm_updates / total_data
    
    def agg_comed(self, agent_updates_dict):
        agent_updates_col_vector = [update.view(-1, 1) for update in agent_updates_dict.values()]
        concat_col_vectors = torch.cat(agent_updates_col_vector, dim=1)
        return torch.median(concat_col_vectors, dim=1).values
    
    def agg_sign(self, agent_updates_dict):
        """ aggregated majority sign update """
        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]
        sm_signs = torch.sign(sum(agent_updates_sign))
        return torch.sign(sm_signs)




