"""Compute baselines for node-classification."""

from collections import defaultdict
import os
import pickle
import sys

import pandas as pd
import torch

sys.path.append("gnnexp")
from models import GCNSynthetic

if len(sys.argv) != 3:
    print("\nUSAGE: python tests/baselines.py [DATASET] [EVAL-MODE]")
    print("DATASET: syn1, syn4, syn5")
    print("EVAL-MODE: train, eval")
    exit(1)

DATASET = sys.argv[1]
EVAL = sys.argv[2]


## ===== Data =====
# Extracted subadjacency matrices.
#todo: automate the output folder path usage.
if DATASET == 'syn1':
    with open("output/syn1/1657970490/original_sub_data.pkl", "rb") as file:
        sub_data = pickle.load(file)
elif DATASET == 'syn4':
    with open("output/syn4/1657890667/original_sub_data.pkl", "rb") as file:
        sub_data = pickle.load(file)

sub_labels = dict()
for node in sub_data:
    new_idx = sub_data[node]['node_idx_new']
    sub_labels[node] = int(sub_data[node]['sub_label'][new_idx])

explanations = dict()
PATH = f"explanation/{DATASET}_top6"
for filename in os.listdir(PATH):
    if 'label' not in filename:
        continue
    explanations[int(filename[4:7])] = pd.read_csv(f"{PATH}/{filename}", header=None).to_numpy()


## ===== Model =====
ckpt = torch.load(f"data/{DATASET}/eval_as_{EVAL}.pt")
cg_dict = ckpt["cg"]
input_dim = cg_dict["feat"].shape[2] 
num_classes = cg_dict["pred"].shape[2]
feat = torch.from_numpy(cg_dict["feat"]).float()
adj = torch.from_numpy(cg_dict["adj"]).float()
label = torch.from_numpy(cg_dict["label"]).long()

with open(f"tests/prog_args_{DATASET}.pkl", "rb") as file:
    prog_args = pickle.load(file)

model = GCNSynthetic(
    nfeat=input_dim,
    nhid=prog_args.hidden_dim,
    nout=prog_args.output_dim,
    nclass=num_classes,
    dropout=0.0,
)
model.load_state_dict(ckpt["model_state"])
model.eval()


## ===== Predictions =====
predictions = dict()
for node in explanations:
    sub_adj = sub_data[node]['org_adj'] - torch.Tensor(explanations[node]).unsqueeze(0)
    sub_adj = sub_adj + torch.eye(sub_adj.size(-1))
    new_idx = sub_data[node]['node_idx_new']
    pred_proba = model(
        sub_data[node]['sub_feat'],
        sub_adj
    ).squeeze(0)
    predictions[node] = int(torch.argmax(pred_proba[new_idx]))
predictions = torch.Tensor(list(predictions.values()))
labels = torch.Tensor(list(sub_labels.values()))


## ===== Fidelity =====
fidelity = 1 - torch.sum(predictions != labels)/labels.size(0)
print("\n====================")
print(f"Fidelity: {fidelity:.2f}")


## ===== Per label fidelity =====
per_label_mismatches = defaultdict(int)
for label, pred in zip(labels, predictions):
    if label == pred:
        continue
    per_label_mismatches[int(label)] += 1
nodes_per_label = {
    int(key):int(val) for key, val in zip(
        labels.unique(return_counts=True)[0],
        labels.unique(return_counts=True)[1]
    )
}
print("\n====================")
print("Per label fidelity:\n")
for label in per_label_mismatches:
    print(f"Label-{label}", end=': ')
    print(f"{1 - per_label_mismatches[label]/nodes_per_label[label]:.2f}")
    print()