import os
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
    model_path = exp_args.model_path
    # train_indices = np.load(
    #     os.path.join(model_path, 'train_indices.pickle'),
    #     allow_pickle=True
    # ) # * We don't need this.
    test_indices = np.load(
        "datasets/NCI1/index.pkl",
        allow_pickle=True
    )
    test_indices = test_indices["idx_test"]
    G_dataset = nci1_preprocessing('datasets/NCI1/raw/')
    graphs = G_dataset.graphs
    labels = G_dataset.labels
    if exp_args.gpu:
        device = torch.device('cuda:%s' % exp_args.cuda)
    else:
        device = 'cpu'
    base_model = GCNGraph(G_dataset.feat_dim, 128).to(device) # todo: We need a new architecture which matches the weight_dictionary.
    base_model.load_state_dict(
        torch.load(os.path.join(model_path, 'model.model'))
    ) # todo: We have to move to weights to this location.
    #  fix the base model
    for param in base_model.parameters():
        param.requires_grad = False

    # Create explainer
    explainer = GraphExplainerEdge(
        base_model=base_model,
        G_dataset=G_dataset,
        args=exp_args,
        test_indices=test_indices,
        # fix_exp=15
    )

    FOLDER_PATH = exp_args.output
    explainer.explain_nodes_gnn_stats(FOLDER_PATH)
