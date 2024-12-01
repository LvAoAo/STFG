import numpy as np
from collections import defaultdict
import logging
import torch


def calculate_sparsities(args, params, tabu=[], distribution="ERK"):
    spasities = {}
    if distribution == "uniform":
        for name in params:
            if name not in tabu:
                spasities[name] = 1 - args.dense_ratio
            else:
                spasities[name] = 0
    elif distribution == "ERK":
        logging.info('initialize by ERK')
        total_params = 0
        for name in params:
            total_params += params[name].numel()
        is_epsilon_valid = False
        # # The following loop will terminate worst case when all masks are in the
        # custom_sparsity_map. This should probably never happen though, since once
        # we have a single variable or more with the same constant, we have a valid
        # epsilon. Note that for each iteration we add at least one variable to the
        # custom_sparsity_map and therefore this while loop should terminate.
        dense_layers = set()

        density = args.dense_ratio
        while not is_epsilon_valid:
            # We will start with all layers and try to find right epsilon. However if
            # any probablity exceeds 1, we will make that layer dense and repeat the
            # process (finding epsilon) with the non-dense layers.
            # We want the total number of connections to be the same. Let say we have
            # for layers with N_1, ..., N_4 parameters each. Let say after some
            # iterations probability of some dense layers (3, 4) exceeded 1 and
            # therefore we added them to the dense_layers set. Those layers will not
            # scale with erdos_renyi, however we need to count them so that target
            # paratemeter count is achieved. See below.
            # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
            #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
            # eps * (p_1 * N_1 + p_2 * N_2) =
            #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
            # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.

            divisor = 0
            rhs = 0
            raw_probabilities = {}
            for name in params:
                if name in tabu or "running" in name or "track" in name :
                    dense_layers.add(name)
                n_param = np.prod(params[name].shape)
                n_zeros = n_param * (1 - density)
                n_ones = n_param * density

                if name in dense_layers:
                    rhs -= n_zeros
                else:
                    rhs += n_ones
                    raw_probabilities[name] = (
                                                      np.sum(params[name].shape) / np.prod(params[name].shape)
                                              ) ** 1
                    divisor += raw_probabilities[name] * n_param
            epsilon = rhs / divisor
            max_prob = np.max(list(raw_probabilities.values()))
            max_prob_one = max_prob * epsilon
            if max_prob_one > 1:
                is_epsilon_valid = False
                for mask_name, mask_raw_prob in raw_probabilities.items():
                    if mask_raw_prob == max_prob:
                        print(f"Sparsity of var:{mask_name} had to be set to 0.")
                        dense_layers.add(mask_name)
            else:
                is_epsilon_valid = True

        # With the valid epsilon, we can set sparsities of the remaning layers.
        for name in params:
            if name in dense_layers:
                spasities[name] = 0
            else:
                spasities[name] = (1 - epsilon * raw_probabilities[name])
    return spasities

def init_masks(params, sparsities):
    masks = {}
    for name in params:
        masks[name] = torch.zeros_like(params[name])
        dense_numel = int((1 - sparsities[name]) * torch.numel(masks[name]))
        if dense_numel > 0:
            temp = masks[name].view(-1)
            perm = torch.randperm(len(temp))
            perm = perm[:dense_numel]
            temp[perm] = 1
        masks[name] = masks[name].to("cpu")
    return masks


def distribute_data_dirichlet(dataset, args):
    # sort labels
    labels_sorted = dataset.targets.sort()
    # create a list of pairs (index, label), i.e., at index we have an instance of  label
    class_by_labels = list(zip(labels_sorted.values.tolist(), labels_sorted.indices.tolist()))
    labels_dict = defaultdict(list)

    for k, v in class_by_labels:
        labels_dict[k].append(v)
    # convert list to a dictionary, e.g., at labels_dict[0], we have indexes for class 0
    N = len(labels_sorted[1])
    K = len(labels_dict)
    logging.info((N, K))
    client_num = args.num_agents

    min_size = 0
    while min_size < 10:
        idx_batch = [[] for _ in range(client_num)]
        for k in labels_dict:
            idx_k = labels_dict[k]

            # get a list of batch indexes which are belong to label k
            np.random.shuffle(idx_k)
            # using dirichlet distribution to determine the unbalanced proportion for each client (client_num in total)
            # e.g., when client_num = 4, proportions = [0.29543505 0.38414498 0.31998781 0.00043216], sum(proportions) = 1
            proportions = np.random.dirichlet(np.repeat(args.alpha, client_num))

            # get the index in idx_k according to the dirichlet distribution
            proportions = np.array([p * (len(idx_j) < N / client_num) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

            # generate the batch list for each client
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    # distribute data to users
    dict_users = defaultdict(list)
    for user_idx in range(args.num_agents):
        dict_users[user_idx] = idx_batch[user_idx]
        np.random.shuffle(dict_users[user_idx])

    # num = [ [ 0 for k in range(K) ] for i in range(client_num)]
    # for k in range(K):
    #     for i in dict_users:
    #         num[i][k] = len(np.intersect1d(dict_users[i], labels_dict[k]))
    # logging.info(num)
    # print(dict_users)
    # def intersection(lst1, lst2):
    #     lst3 = [value for value in lst1 if value in lst2]
    #     return lst3
    # # logging.info( [len(intersection (dict_users[i], dict_users[i+1] )) for i in range(args.num_agents)] )
    return dict_users

def vector_to_model(vec, model):
    # Pointer for slicing the vector for each parameter
    state_dict = model.state_dict()
    pointer = 0
    for name in state_dict:
        # The length of the parameter
        num_param = state_dict[name].numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        state_dict[name].data = vec[pointer:pointer + num_param].view_as(state_dict[name]).data
        # Increment the pointer
        pointer += num_param
    model.load_state_dict(state_dict)
    return state_dict

def vector_to_name_param(vec, name_param_map):
    pointer = 0
    for name in name_param_map:
        # The length of the parameter
        num_param = name_param_map[name].numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        name_param_map[name].data = vec[pointer:pointer + num_param].view_as(name_param_map[name]).data
        # Increment the pointer
        pointer += num_param

    return name_param_map
