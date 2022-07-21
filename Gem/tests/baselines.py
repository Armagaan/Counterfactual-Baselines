# Baselines
from collections import defaultdict
import os
import pickle
import sys

import pandas as pd
import torch

sys.path.append("gnnexp")
from models import GCNSynthetic

if len(sys.argv) != 4:
    print("\nUSAGE: python tests/baselines.py [DATASET] [EVALMODE] [OUTPUTS-FOLDER]")
    print("[DATASET]: syn1, syn4, syn5")
    print("[EVALMODE]: train, eval\n")
    print("[OUTPUTS-FOLDER]: path to folder containing the output of evaluate_adj.py")
    print("Hint: output/syn1/12345, output/syn4/12345, output/syn5/12345")
    exit(1)

## ===== CONSTANTS =====
DATASET = sys.argv[1]
EVAL = sys.argv[2]
OUTPUTS = sys.argv[3]

if  DATASET not in ['syn1', 'syn4', 'syn5']:
    print("INVALID DATASET!")
    exit(1)
elif EVAL not in ['eval', 'train']:
    print("INVALID EVALMODE!")
    exit(1)

## ===== DATA =====
# The extracted subadjacency matrices.
with open(f"{OUTPUTS}/original_sub_data.pkl", "rb") as file:
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
    node_idx = ''.join(filter(lambda i: i.isdigit(), filename))
    explanations[int(node_idx)] = pd.read_csv(f"{PATH}/{filename}", header=None).to_numpy()


## ===== MODEL =====
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


## ===== PREDICTIONS =====
predictions = dict()
for node in explanations:
    sub_adj = sub_data[node]['org_adj'] - torch.Tensor(explanations[node]).unsqueeze(0)
    new_idx = sub_data[node]['node_idx_new']
    pred_proba = model(
        sub_data[node]['sub_feat'],
        sub_adj
    ).squeeze(0)
    predictions[node] = int(torch.argmax(pred_proba[new_idx]))


## ===== FIDELITY =====
misclassifications = 0
for node in predictions:
    if predictions[node] != sub_labels[node]:
        misclassifications += 1
fidelity = 1 - misclassifications/len(predictions)
print("\n===============")
print(f"Fidelity: {fidelity:.2f}")


## ===== PER-LABEL FIDELITY =====
per_label_mismatches = defaultdict(int)
for node in predictions:
    label = sub_labels[node]
    if predictions[node] != label:
        per_label_mismatches[int(label)] += 1
labels, label_counts = torch.Tensor(list(sub_labels.values())).unique(return_counts=True)
nodes_per_label = {
    int(key):int(val) for key, val in zip(labels, label_counts)
}
print("\n===============")
print("Per label fidelity:")
for label in per_label_mismatches:
    print(f"Label-{label}", end=": ")
    print(f"{1 - per_label_mismatches[label]/nodes_per_label[label]:.2f}")
    print()
