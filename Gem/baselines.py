# Baselines
import argparse
import os
import pickle
import sys

import torch
import numpy as np

sys.path.append("gnnexp")
import models
import utils.io_utils as io_utils
import utils.parser_utils as parser_utils


## Data

## Model
def arg_parse():
    parser = argparse.ArgumentParser(description="GNN Explainer arguments.")
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument("--dataset", dest="dataset", help="Input dataset.")
    benchmark_parser = io_parser.add_argument_group()
    benchmark_parser.add_argument(
        "--bmname", dest="bmname", help="Name of the benchmark dataset"
    )
    io_parser.add_argument("--pkl", dest="pkl_fname", help="Name of the pkl data file")

    parser_utils.parse_optimizer(parser)

    parser.add_argument("--clean-log", action="store_true", help="If true, cleans the specified log directory before running.")
    parser.add_argument("--logdir", dest="logdir", help="Tensorboard log directory")
    parser.add_argument("--ckptdir", dest="ckptdir", help="Model checkpoint directory")
    parser.add_argument("--cuda", dest="cuda", help="CUDA.")
    parser.add_argument(
        "--gpu",
        dest="gpu",
        action="store_const",
        const=True,
        default=False,
        help="whether to use GPU.",
    )
    parser.add_argument(
        "--epochs", dest="num_epochs", type=int, help="Number of epochs to train."
    )
    parser.add_argument(
        "--hidden-dim", dest="hidden_dim", type=int, help="Hidden dimension"
    )
    parser.add_argument(
        "--output-dim", dest="output_dim", type=int, help="Output dimension"
    )
    parser.add_argument(
        "--num-gc-layers",
        dest="num_gc_layers",
        type=int,
        help="Number of graph convolution layers before each pooling",
    )
    parser.add_argument(
        "--bn",
        dest="bn",
        action="store_const",
        const=True,
        default=False,
        help="Whether batch normalization is used",
    )
    parser.add_argument("--dropout", dest="dropout", type=float, help="Dropout rate.")
    parser.add_argument(
        "--nobias",
        dest="bias",
        action="store_const",
        const=False,
        default=True,
        help="Whether to add bias. Default to True.",
    )
    parser.add_argument(
        "--no-writer",
        dest="writer",
        action="store_const",
        const=False,
        default=True,
        help="Whether to do SummaryWriter. Default to True.",
    )
    # Explainer
    parser.add_argument("--mask-act", dest="mask_act", type=str, help="sigmoid, ReLU.")
    parser.add_argument(
        "--mask-bias",
        dest="mask_bias",
        action="store_const",
        const=True,
        default=False,
        help="Whether to add mask bias. Default to True.",
    )
    parser.add_argument(
        "--explain-node", dest="explain_node", type=int, help="Node to explain."
    )
    parser.add_argument(
        "--graph-idx", dest="graph_idx", type=int, help="Graph to explain."
    )
    parser.add_argument(
        "--graph-mode",
        dest="graph_mode",
        action="store_const",
        const=True,
        default=False,
        help="whether to run Explainer on Graph Classification task.",
    )
    parser.add_argument(
        "--multigraph-class",
        dest="multigraph_class",
        type=int,
        help="whether to run Explainer on multiple Graphs from the Classification task for examples in the same class.",
    )
    parser.add_argument(
        "--multinode-class",
        dest="multinode_class",
        type=int,
        help="whether to run Explainer on multiple nodes from the Classification task for examples in the same class.",
    )
    parser.add_argument(
        "--align-steps",
        dest="align_steps",
        type=int,
        help="Number of iterations to find P, the alignment matrix.",
    )

    parser.add_argument(
        "--method", dest="method", type=str, help="Method. Possible values: base, att."
    )
    parser.add_argument(
        "--name-suffix", dest="name_suffix", help="suffix added to the output filename"
    )
    parser.add_argument(
        "--explainer-suffix",
        dest="explainer_suffix",
        help="suffix added to the explainer log",
    )
    parser.add_argument('--n_hops', type=int, default=3, help='Number of hops.')
    parser.add_argument('--top_k', type=int, default=None, help='Keep k edges of prediction.')
    parser.add_argument('--threshold', type=float, default=None, help='Keep k edges of prediction.')
    parser.add_argument('--output', type=str, default=None, help='output path.')

    # TODO: Check argument usage
    parser.set_defaults(
        logdir="log",
        ckptdir="ckpt",
        dataset="Mutagenicity",
        opt="adam",  
        opt_scheduler="none",
        cuda="cuda:0",
        lr=0.1,
        clip=2.0,
        batch_size=20,
        num_epochs=100,
        hidden_dim=20,
        output_dim=20,
        num_gc_layers=3,
        dropout=0.0,
        method="base",
        name_suffix="",
        explainer_suffix="",
        align_steps=1000,
        explain_node=None,
        graph_idx=-1,
        graph_mode=False,
        mask_act="sigmoid",
        multigraph_class=-1,
        multinode_class=-1,
    )
    return parser.parse_args()
# sys.argv.append("--dataset=syn1")
# sys.argv.append("--top_k=6")

prog_args = arg_parse()


# bashapes
if prog_args.dataset == "syn1":
    FOLDER = "output/syn1/1656673538"
# treecycles
elif prog_args.dataset == "syn4":
    FOLDER = "output/syn4/1656673532"
with open(f"{FOLDER}/original.pkl", "rb") as file:
    original_data = pickle.load(file)
with open(f"{FOLDER}/gem.pkl", "rb") as file:
    gem_data = pickle.load(file)


if prog_args.output is None:
    prog_args.output = prog_args.dataset
if prog_args.top_k is not None:
    prog_args.output += '_top%d' % (prog_args.top_k)
elif prog_args.threshold is not None:
    prog_args.output += '_threshold%s' % (prog_args.threshold)
os.makedirs("distillation/%s" %prog_args.output, exist_ok=True)

device = torch.device(prog_args.cuda if prog_args.gpu and torch.cuda.is_available() else "cpu")
if prog_args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = prog_args.cuda
    print("CUDA", prog_args.cuda)
else:
    print("Using CPU")

# Load a model checkpoint
ckpt = io_utils.load_ckpt(prog_args)
cg_dict = ckpt["cg"] # get computation graph
input_dim = cg_dict["feat"].shape[2] 
num_classes = cg_dict["pred"].shape[2]
print("Loaded model from {}".format(prog_args.ckptdir))
print("input dim: ", input_dim, "; num classes: ", num_classes)
model = models.GcnEncoderNode(
    input_dim=input_dim,
    hidden_dim=prog_args.hidden_dim,
    embedding_dim=prog_args.output_dim,
    label_dim=num_classes,
    num_layers=prog_args.num_gc_layers,
    bn=prog_args.bn,
    args=prog_args,
)
model.load_state_dict(ckpt["model_state"])
model.eval()
# feat = torch.from_numpy(cg_dict["feat"]).float()
# adj = torch.from_numpy(cg_dict["adj"]).float()
label = torch.from_numpy(cg_dict["label"]).long()
# preds, _ = model(feat, adj)


## Predict
original_labels = list()
original_predictions = list()
gem_predictions = list()
print("label\tpred\tgem")
for orig, gem in zip(original_data, gem_data):
    node_id = orig['new_id']
    org_adj = orig['adj']
    org_feat = orig['feat']
    gem_adj = gem['adj']
    gem_feat = gem['feat']
    perturbed_adj = org_adj - gem_adj
    feat = gem_feat
    org_probas, __ = model(feat, org_adj)
    gem_probas, __ = model(feat, perturbed_adj)
    org_prediction = int(torch.argmax(org_probas[0][node_id]))
    gem_prediction = int(torch.argmax(gem_probas[0][node_id]))
    original_label = label[0][orig['id']]

    original_labels.append(original_label)
    original_predictions.append(org_prediction)
    gem_predictions.append(gem_prediction)
    print(f"{original_label}\t{org_prediction}\t{gem_prediction}")

print(
    f"Accuracy: ",
    f"{torch.sum(torch.Tensor(original_predictions) == torch.Tensor(original_labels))/len(original_labels):.2f}"
)

print(
    f"Fidelity: ",
    f"{1 - torch.sum(torch.Tensor(original_predictions) != torch.Tensor(gem_predictions))/len(original_predictions):.2f}"
)