
## * Imports
from collections import defaultdict
import numpy as np
import pickle
import sys

from torch import sigmoid

if len(sys.argv) != 2:
    print(f"Usage: python tests.py [FOLDER]")
    print("FOLDER: Specify the folder containing the outputs of get_outputs.sh.")
    exit(1)

# Specify the folder containing the outputs of get_outputs.sh.
FOLDER = sys.argv[1]

with open(f"{FOLDER}/exp_dict.pkl", "rb") as file:
    exp_dict = pickle.load(file) # format: node_id: explanation_mask over the adjacency_matrix

with open(f"{FOLDER}/log.txt", "r") as file:
    log = file.readlines()

with open(f"{FOLDER}/num_dict.pkl", "rb") as file:
    num_dict = pickle.load(file) # format: node_id: #counterfactuals_found

with open(f"{FOLDER}/pred_label_dict.pkl", "rb") as file:
    pred_label_dict = pickle.load(file) # format: node_id: initial_blackbox_prediction

with open(f"{FOLDER}/pred_proba.txt", "r") as file:
    pred_proba = file.readlines()

with open(f"{FOLDER}/t_gid.pkl", "rb") as file:
    t_gid = pickle.load(file) # format: subgraph_id (same as node_id)


## * Constants
NUMBER_OF_LABELS = 2 
# ! Temporary:
# ! There should be 2 labels. But, since cfsqr is explaining label 1,
# ! there is no graph with label 0 in the set. 
# NUMBER_OF_LABELS = len(str.strip(pred_proba[0]).split())

NODES_PER_LABEL = defaultdict(int)
for node_id, label in pred_label_dict.items():
    NODES_PER_LABEL[f"label-{int(label)}"] += 1


## * Per-label Explanation size
per_label_explanation_size = defaultdict(list)

# iterate over the nodes
for node_id, number_of_cfs in num_dict.items():
    # find out the initial label
    label = pred_label_dict[node_id]
    # update size of corresponding label
    per_label_explanation_size[f"label-{int(label)}"].append(int(number_of_cfs))

# find mean and std
for label in range(NUMBER_OF_LABELS):
    if len(per_label_explanation_size[f"label-{int(label)}"]) == 0:
        mean, std = None, None
    else:
        mean = np.mean(per_label_explanation_size[f"label-{int(label)}"])
        std = np.std(per_label_explanation_size[f"label-{int(label)}"])
    per_label_explanation_size[f"label-{int(label)}"] = [mean, std]
print("\n###############\n")
print("Per Label Explanation size:")
for key, value in per_label_explanation_size.items(): # format: label: (mean, std)
    print(f"{key}: {value[0]} +- {value[1]}")


## * Explanation size
mean = np.array(list(num_dict.values())).mean()
std = np.array(list(num_dict.values())).std()
explanation_size = [mean, std]
print("\n###############\n")
print(f"Explanation size: {explanation_size[0]:.2f} +- {explanation_size[1]:.2f}")


## * Per-node fidelity
predictions = defaultdict(int)
for node_id, line in zip(t_gid, pred_proba):
    line = line.strip().split()
    line = [float(pred) for pred in line]
    sigmoid_proba = line[0]
    predictions[node_id] = round(sigmoid_proba)
labels_and_preds = defaultdict(tuple)

for node_id in t_gid:
    labels_and_preds[node_id] = (int(pred_label_dict[node_id]), predictions[node_id])
per_label_cf_found = defaultdict(int)

for node_id, (label, prediction) in labels_and_preds.items():
    if label != prediction:
        per_label_cf_found[f"label-{label}"] += 1

per_label_fidelity = dict()
for key, value in per_label_cf_found.items():
    per_label_fidelity[key] = 1 - per_label_cf_found[key]/NODES_PER_LABEL[key]
print("\n###############\n")
print(f"Per Label Fidelity:\n")
for key, value in per_label_fidelity.items():
    print(f"{key}: {value}")


## * Fidelity
cf_found = 0
for node_id, (label, prediction) in labels_and_preds.items():
    if label != prediction:
        cf_found += 1

fidelity = 1 - cf_found/sum(list(NODES_PER_LABEL.values()))
print("\n###############\n")
print(f"Fidelity: {fidelity}")
