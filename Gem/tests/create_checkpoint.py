"""Create a checkpoint for use in gem's scripts."""

import os
import pickle
import sys

import torch

if len(sys.argv) != 2:
    print("\nUSAGE: python tests/create_checkpoint.py [DATASET]")
    print("[DATASET]: syn1, syn4, syn5")
    exit(1)


## ===== CONSTANTS =====
DATASET = sys.argv[1]


## ===== DATA =====
with torch.no_grad():
    ckpt = torch.load(f"ckpt/{DATASET}_base_h20_o20.pth.tar")
    cg_dict = ckpt["cg"] # Get the graph data.
    input_dim = cg_dict["feat"].shape[2]
    adj = cg_dict["adj"][0]
    label = cg_dict["label"][0]
    features = torch.tensor(cg_dict["feat"][0], dtype=torch.float)
    num_class = max(label)+1


## ===== CFGNN MODEL WEIGHTS =====
state_dict_cfgnn = torch.load(f"cfgnn_model_weights/gcn_3layer_{DATASET}.pt")


## ===== EVAL SET =====
with open("../eval_set.pkl", "rb") as file:
    eval_set = pickle.load(file)

if DATASET == "syn1":
    KEY = "syn1/ba-shapes"
elif DATASET == "syn4":
    KEY = "syn4/tree-cycles"
elif DATASET == "syn5":
    KEY = "syn5/tree-grids"


# Our eval set as part of the training set
train_set_indices = [range(label.shape[0])]
test_set_indices = eval_set[KEY]
# Save
ckpt["cg"]["train_idx"] = train_set_indices
ckpt["cg"]["test_idx"] = test_set_indices
ckpt["model_state"] = state_dict_cfgnn
os.makedirs(f"data/{DATASET}", exist_ok=True)
torch.save(ckpt, f"data/{DATASET}/eval_as_train.pt")


# Our eval set as the validation set
train_set_indices = [i for i in range(label.shape[0]) if i not in eval_set[KEY]]
test_set_indices = eval_set[KEY]
# Save
ckpt["cg"]["train_idx"] = train_set_indices
ckpt["cg"]["test_idx"] = test_set_indices
ckpt["model_state"] = state_dict_cfgnn
os.makedirs(f"data/{DATASET}", exist_ok=True)
torch.save(ckpt, f"data/{DATASET}/eval_as_eval.pt")
