import os
import pickle
import sys

import numpy as np
import torch

from models.explainer_models import GraphExplainerEdge
from models.gcn import GCNGraph
from utils.argument import arg_parse_exp_graph_nci1
from utils.preprocessing.nci1_preprocessing import nci1_preprocessing


if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    torch.manual_seed(0)
    np.random.seed(0)
    exp_args = arg_parse_exp_graph_nci1()
    print("argument:\n", exp_args)
    
    with open("datasets/Eval-sets/indices_nci1.pkl", "rb") as file:
        indices = pickle.load(file)
    train_indices = indices['idx_train'].numpy()
    val_indices = indices['idx_val'].numpy()
    test_indices = indices['idx_test'].numpy()
    
    G_dataset = nci1_preprocessing('datasets/NCI1/raw/')
    graphs = G_dataset.graphs
    labels = G_dataset.labels
    if exp_args.gpu:
        device = torch.device('cuda:%s' % exp_args.cuda)
    else:
        device = 'cpu'
    base_model = GCNGraph(G_dataset.feat_dim, 128).to(device)
    state_dict = torch.load("graph_classification_model_weights/new_nci1_weights.pt")
    base_model.load_state_dict(state_dict)
    #  fix the base model
    for param in base_model.parameters():
        param.requires_grad = False

    # Create explainer
    explainer = GraphExplainerEdge(
        base_model=base_model,
        G_dataset=G_dataset,
        args=exp_args,
        test_indices=val_indices,
        # fix_exp=15
    )

    FOLDER_PATH = exp_args.output
    explainer.explain_nodes_gnn_stats(FOLDER_PATH)
