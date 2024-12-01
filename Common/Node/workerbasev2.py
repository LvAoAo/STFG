import logging
import torch
import torch.nn as nn
import math
import time
from abc import ABCMeta, abstractmethod
import numpy as np
from torch._C import device
import torch.nn.functional as F
from GNN_common.train.metrics import accuracy_TU as accuracy
import os
import ipdb
from torch.nn.utils import vector_to_parameters, parameters_to_vector
logger = logging.getLogger('client.workerbase')



'''
This is the worker for sharing the local weights.
'''
class WorkerBaseV2(metaclass=ABCMeta):
    def __init__(self, model, loss_func, train_iter, attack_iter, test_iter, config, optimizer, device):
        self.model = model
        self.loss_func = loss_func

        self.train_iter = train_iter
        self.test_iter = test_iter
        self.attack_iter = attack_iter
        self.config = config
        self.optimizer = optimizer

        # Accuracy record
        self.acc_record = [0]

        self.device = device
        self._level_length = None
        self._weights_len = 0
        self._weights = None
        self._weights_list = None
        self._update = None

    def get_weights(self):
        """ getting weights """
        return self._weights
    
    def get_weights_list(self):
        """ getting weights as list """
        return self._weights_list

    def set_weights(self, weights):
        """ setting weights """
        self._weights = weights

    def get_update(self):
        return self._update

    def upgrade(self):
        """ Use the processed weights to update the model """
        self.model.load_state_dict(self._weights)

    @abstractmethod
    def update(self):
        pass

    ## GNN model training:

    def gnn_train_v2(self): # This function is for local train one epoch using local dataset on client
        """ General local training methods """
        initial_model_params = parameters_to_vector(self.model.parameters()).detach()
        self.model.train()
        self.acc_record = [0]
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for batch_graphs, batch_labels in self.train_iter:
            batch_graphs = batch_graphs.to(self.device)
            batch_x = batch_graphs.ndata['feat'].to(self.device)  # num x feat
            batch_e = batch_graphs.edata['feat'].to(self.device)
            batch_labels = batch_labels.to(torch.long)
            batch_labels = batch_labels.to(self.device)
            batch_scores = self.model.forward(batch_graphs, batch_x, batch_e)
            l = self.model.loss(batch_scores, batch_labels)
            self.optimizer.zero_grad()
            l.backward()
            self.optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += accuracy(batch_scores, batch_labels)
            n += batch_labels.size(0)
            batch_count += 1

        self._weights_list = []
        self._level_length = [0]
        
        for param in self.model.parameters():
            self._level_length.append(param.data.numel() + self._level_length[-1])
            self._weights_list += param.data.view(-1).cpu().numpy().tolist()

        self._weights = self.model.state_dict()

        # print train acc of each client
        if self.attack_iter is not None:
            test_acc, test_l,  att_acc = self.gnn_evaluate()
        else:
            test_acc, test_l = self.gnn_evaluate()
        with torch.no_grad():
            self._update = parameters_to_vector(self.model.parameters()).double() - initial_model_params
        return train_l_sum / batch_count, train_acc_sum / n, test_l, test_acc

    def gnn_evaluate(self):
        acc_sum, acc_att, n, test_l_sum = 0.0, 0.0, 0, 0.0
        batch_count = 0
        for batch_graphs, batch_labels in self.test_iter:
            batch_graphs = batch_graphs.to(self.device)
            self.model.eval()
            batch_x = batch_graphs.ndata['feat'].to(self.device)
            batch_e = batch_graphs.edata['feat'].to(self.device)
            batch_labels = batch_labels.to(torch.long)
            batch_labels = batch_labels.to(self.device)
    
            batch_scores = self.model.forward(batch_graphs, batch_x, batch_e)
            l = self.loss_func(batch_scores, batch_labels)
            acc_sum += accuracy(batch_scores, batch_labels)
            test_l_sum += l.detach().item()
            n += batch_labels.size(0)
            batch_count += 1
            if self.attack_iter is not None:
                n_att = 0
                for batch_graphs, batch_labels in self.attack_iter:
                    batch_graphs = batch_graphs.to(self.device)
                    self.model.eval()
                    batch_x = batch_graphs.ndata['feat'].to(self.device)
                    batch_e = batch_graphs.edata['feat'].to(self.device)
                    batch_labels = batch_labels.to(self.device)

                    batch_scores = self.model.forward(batch_graphs, batch_x, batch_e)
                    acc_att += accuracy(batch_scores, batch_labels)
                    self.model.train()
                    n_att += batch_labels.size(0)
                return acc_sum / n, test_l_sum / batch_count, acc_att / n_att

        return acc_sum / n, test_l_sum / batch_count
 
class ClearDenseClient(WorkerBaseV2):
    def __init__(self, client_id, model, loss_func, train_iter, attack_iter, test_iter, config, optimizer, device, grad_stub, args, scheduler):
        super(ClearDenseClient, self).__init__(
            model=model, 
            loss_func=loss_func, 
            train_iter=train_iter, 
            attack_iter=attack_iter, 
            test_iter=test_iter, 
            config=config, 
            optimizer=optimizer, 
            device=device
        )
        self.client_id = client_id
        self.grad_stub = None
        self.args = args
        self.scheduler = scheduler

    def update(self):
        pass

class ClearSparseClient(WorkerBaseV2):
    def __init__(self, client_id, model, loss_func, train_iter, attack_iter, test_iter, config, optimizer, device, grad_stub, args, scheduler, mask):
        super(ClearSparseClient, self).__init__(
            model=model, 
            loss_func=loss_func, 
            train_iter=train_iter, 
            attack_iter=attack_iter, 
            test_iter=test_iter, 
            config=config, 
            optimizer=optimizer, 
            device=device
        )
        self.client_id = client_id
        self.grad_stub = None
        self.args = args
        self.scheduler = scheduler
        self.mask = mask
        self.num_remove = None
    
    def gnn_train_v2(self, round=None): # This function is for local train one epoch using local dataset on client
        """ General local training methods """
        initial_model_params = parameters_to_vector([ self.model.state_dict()[name] for name in self.model.state_dict()]).detach()
        self.model.train()
        for name, param in self.model.named_parameters():
            self.mask[name] =self.mask[name].to(self.device)
            param.data = param.data * self.mask[name]
        if self.num_remove != None:
            # add mask according to gradient
            gradient = self.screen_gradients(self.model)
            self.mask = self.update_mask(self.mask, self.num_remove, gradient)
        self.acc_record = [0]
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        # Warning: No local epochs
        for batch_graphs, batch_labels in self.train_iter:
            batch_graphs = batch_graphs.to(self.device)
            batch_x = batch_graphs.ndata['feat'].to(self.device)  # num x feat
            batch_e = batch_graphs.edata['feat'].to(self.device)
            batch_labels = batch_labels.to(torch.long)
            batch_labels = batch_labels.to(self.device)
            batch_scores = self.model.forward(batch_graphs, batch_x, batch_e)
            l = self.model.loss(batch_scores, batch_labels)
            self.optimizer.zero_grad()
            l.backward()
            for name, param in self.model.named_parameters():
                param.grad.data = self.mask[name].to(self.device) * param.grad.data
            self.optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += accuracy(batch_scores, batch_labels)
            n += batch_labels.size(0)
            batch_count += 1

        self.mask, self.num_remove = self.fire_mask(self.model.state_dict(), self.mask, round)

        with torch.no_grad():
            after_train = parameters_to_vector([ self.model.state_dict()[name] for name in self.model.state_dict()]).detach()
            array_mask = parameters_to_vector([ self.mask[name].to(self.device) for name in self.model.state_dict()]).detach()
            self._update = ( array_mask *(after_train - initial_model_params))
            # if "scale" in self.args.attack:
            #     logging.info("scale update for" + self.args.attack.split("_",1)[1] + " times")
            #     if self.id<  self.args.num_corrupt:
            #         self.update=  int(self.args.attack.split("_",1)[1]) * self.update
        for name, param in self.model.named_parameters():
            self.mask[name] =self.mask[name].to(self.device)
            param.data = param.data * self.mask[name]
        
        self._weights_list = []
        self._level_length = [0]
        for param in self.model.parameters():
            self._level_length.append(param.data.numel() + self._level_length[-1])
            self._weights_list += param.data.view(-1).cpu().numpy().tolist()

        self._weights = self.model.state_dict()

        # print train acc of each client
        if self.attack_iter is not None:
            test_acc, test_l,  att_acc = self.gnn_evaluate()
        else:
            test_acc, test_l = self.gnn_evaluate()

        return train_l_sum / batch_count, train_acc_sum / n, test_l, test_acc
    
    def update(self):
        pass


    def screen_gradients(self, model):
        model.train()
        # # # train and update
        criterion = nn.CrossEntropyLoss()
        gradient = {name: 0 for name, param in model.named_parameters()}
        # # sample 10 batch  of data
        batch_num = 0
        for batch_graphs, batch_labels in self.train_iter:
            batch_graphs = batch_graphs.to(self.device)
            batch_x = batch_graphs.ndata['feat'].to(self.device)  # num x feat
            batch_e = batch_graphs.edata['feat'].to(self.device)
            batch_labels = batch_labels.to(torch.long)
            batch_labels = batch_labels.to(self.device)
            batch_scores = self.model.forward(batch_graphs, batch_x, batch_e)
            l = self.model.loss(batch_scores, batch_labels)
            l.backward()
            for name, param in model.named_parameters():
                gradient[name] += param.grad.data
        return gradient

    def update_mask(self, masks, num_remove, gradient=None):
        for name in gradient:
            temp = torch.where(masks[name].to(self.device) == 0, torch.abs(gradient[name]),
                                -100000 * torch.ones_like(gradient[name]))
            sort_temp, idx = torch.sort(temp.view(-1), descending=True)
            masks[name].view(-1)[idx[:num_remove[name]]] = 1
        return masks
    
    # def init_mask(self,  gradient=None):
    #     for name in self.mask:
    #         num_init = torch.count_nonzero(self.mask[name])
    #         self.mask[name] = torch.zeros_like(self.mask[name])
    #         sort_temp, idx = torch.sort(torch.abs(gradient[name]).view(-1), descending=True)
    #         self.mask[name].view(-1)[idx[:num_init]] = 1
             

    def fire_mask(self, weights, masks, round):
        
        drop_ratio = self.args.anneal_factor / 2 * (1 + np.cos((round * np.pi) / (self.args.epochs)))
    
        # logging.info(drop_ratio)
        num_remove = {}
        for name in masks:
                num_non_zeros = torch.sum(masks[name].to(self.device))
                num_remove[name] = math.ceil(drop_ratio * num_non_zeros)
     
        for name in masks:
            if num_remove[name]>0 and  "track" not in name and "running" not in name: 
                temp_weights = torch.where(masks[name].to(self.device) > 0, torch.abs(weights[name]),
                                        100000 * torch.ones_like(weights[name]))
                x, idx = torch.sort(temp_weights.view(-1).to(self.device))
                masks[name].view(-1)[idx[:num_remove[name]]] = 0
        return masks, num_remove


