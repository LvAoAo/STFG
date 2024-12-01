#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python clean_fedgnn.py \
    --dataset NCI1 \
    --config ./GNN_common/configs/TUS/TUs_graph_classification_GCN_NCI1_100k.json \
    --num_workers 5 \
    --num_mali 0 \
    --filename ./Results/Clean

CUDA_VISIBLE_DEVICES=1 python dis_bkd_fedgnn.py \
    --dataset NCI1 \
    --config ./GNN_common/configs/TUS/TUs_graph_classification_GCN_NCI1_100k.json \
    --num_workers 5 \
    --num_mali 2 \
    --filename ./Results/DBA

CUDA_VISIBLE_DEVICES=2 python cen_bkd_fedgnn.py \
    --dataset NCI1 \
    --config ./GNN_common/configs/TUS/TUs_graph_classification_GCN_NCI1_100k.json \
    --num_workers 5 \
    --num_mali 2 \
    --filename ./Results/CBA

CUDA_VISIBLE_DEVICES=3 python dis_bkd_fedgnn.py \
    --defense foolsgold \
    --dataset NCI1 \
    --config ./GNN_common/configs/TUS/TUs_graph_classification_GCN_NCI1_100k.json \
    --num_workers 5 \
    --num_mali 2 \
    --filename ./Results/DBA_foolsgold

CUDA_VISIBLE_DEVICES=2 python dis_bkd_fedgnn.py \
    --defense lockdown \
    --anneal_factor 0.0005 \
    --dense_ratio 0.75 \
    --dataset NCI1 \
    --config ./GNN_common/configs/TUS/TUs_graph_classification_GCN_NCI1_100k.json \
    --num_workers 5 \
    --num_mali 2 \
    --theta 3 \
    --filename ./Results/DBA_lockdown


# Lockdown
CUDA_VISIBLE_DEVICES=0 python dis_bkd_fedgnn.py \
    --defense lockdown \
    --anneal_factor 0.0005 \
    --dense_ratio 0.25 \
    --dataset NCI1 \
    --config ./GNN_common/configs/TUS/TUs_graph_classification_GAT_NCI1_100k.json \
    --num_workers 5 \
    --num_mali 2 \
    --theta 3 \
    --filename ./Results/DBA_lockdown

CUDA_VISIBLE_DEVICES=1 python dis_bkd_fedgnn.py \
    --dataset NCI1 \
    --config ./GNN_common/configs/TUS/TUs_graph_classification_GAT_NCI1_100k.json \
    --num_workers 5 \
    --num_mali 2 \
    --theta 4 \
    --filename ./Results/DBA_GAT

CUDA_VISIBLE_DEVICES=0 python dis_bkd_fedgnn.py \
    --defense lockdown \
    --anneal_factor 0.0005 \
    --dense_ratio 0.75 \
    --dataset NCI1 \
    --config ./GNN_common/configs/TUS/TUs_graph_classification_GraphSage_NCI1_100k.json \
    --num_workers 5 \
    --num_mali 2 \
    --theta 3 \
    --filename ./Results/DBA_lockdown_Sage

CUDA_VISIBLE_DEVICES=1 python dis_bkd_fedgnn.py \
    --defense lockdown \
    --anneal_factor 0.0005 \
    --dense_ratio 0.75 \
    --dataset TRIANGLES \
    --config ./GNN_common/configs/TUS/TUs_graph_classification_GAT_TRIANGLES_100k.json \
    --num_workers 10 \
    --num_mali 4 \
    --theta 8 \
    --filename ./Results/DBA_TRIANGLES_lockdown_GAT

CUDA_VISIBLE_DEVICES=0 python dis_bkd_fedgnn.py \
    --anneal_factor 0.0005 \
    --dense_ratio 0.75 \
    --dataset TRIANGLES \
    --config ./GNN_common/configs/TUS/TUs_graph_classification_GCN_TRIANGLES_100k.json \
    --num_workers 10 \
    --num_mali 4 \
    --theta 8 \
    --filename ./Results/DBA_TRIANGLES_No_defense_GCN

CUDA_VISIBLE_DEVICES=1 python dis_bkd_fedgnn.py \
    --defense lockdown \
    --anneal_factor 0.0005 \
    --dense_ratio 0.5 \
    --dataset TRIANGLES \
    --config ./GNN_common/configs/TUS/TUs_graph_classification_GAT_TRIANGLES_100k.json \
    --num_workers 10 \
    --num_mali 4 \
    --theta 8 \
    --filename ./Results/DBA_TRIANGLES_lockdown_GAT

CUDA_VISIBLE_DEVICES=1 python dis_bkd_fedgnn.py \
    --defense lockdown \
    --anneal_factor 0.0005 \
    --dense_ratio 0.5 \
    --dataset NCI1 \
    --config ./GNN_common/configs/TUS/TUs_graph_classification_GAT_Larger_NCI1_100k.json \
    --num_workers 5 \
    --num_mali 2 \
    --theta 3 \
    --filename ./Results/DBA_NCI1_lockdown_GAT

CUDA_VISIBLE_DEVICES=3 python dis_bkd_fedgnn.py \
    --anneal_factor 0.0005 \
    --dense_ratio 0.5 \
    --dataset NCI1 \
    --config ./GNN_common/configs/TUS/TUs_graph_classification_GAT_Larger_NCI1_100k.json \
    --num_workers 5 \
    --num_mali 2 \
    --theta 3 \
    --filename ./Results/DBA_NCI1_No_defense_GAT