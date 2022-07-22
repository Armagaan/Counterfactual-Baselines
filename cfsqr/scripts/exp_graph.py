import os
import pickle
import sys

import numpy as np
import torch

from models.explainer_models import GraphExplainerEdge
from models.gcn import GCNGraph_Mutag
from utils.argument import arg_parse_exp_graph_mutag_0
from utils.preprocessing.mutag_preprocessing_0 import mutag_preprocessing_0


if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    torch.manual_seed(0)
    np.random.seed(0)
    exp_args = arg_parse_exp_graph_mutag_0()
    print("argument:\n", exp_args)
    
    print(os.getcwd())
    with open("datasets/Eval-sets/indices_mutagenicity.pkl", "rb") as file:
        indices = pickle.load(file)
    train_indices = indices['idx_train'].numpy()
    val_indices = indices['idx_val'].numpy()
    test_indices = indices['idx_test'].numpy()

    G_dataset = mutag_preprocessing_0(dataset_dir="datasets/Mutagenicity_0")
    graphs = G_dataset.graphs
    labels = G_dataset.labels
    if exp_args.gpu:
        device = torch.device('cuda:%s' % exp_args.cuda)
    else:
        device = 'cpu'
    
    base_model = GCNGraph_Mutag(G_dataset.feat_dim, 128).to(device)
    state_dict = torch.load("graph_classification_model_weights/mutag_weights.pt")
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
