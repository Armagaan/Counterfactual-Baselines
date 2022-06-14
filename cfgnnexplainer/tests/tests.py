# Baselines

## Imports
import pickle
import numpy as np
from collections import defaultdict

import sys
if len(sys.argv) != 2:
    print("USAGE: python tests.py [dataset]")
    print("dataset: one of [bashapes, treecycles, treegrids]")
    exit(1)

# specify the datset
DATASET= sys.argv[1]
if DATASET == "bashapes":
    path_log = "outputs/bashapes/1653496990/log.txt"
    path_cfs = "results/syn1/SGD/syn1_cf_examples_lr0.01_beta0.5_mom0.9_epochs500"
    path_predictions = "outputs/bashapes/1653496990/predictions.pkl"
    path_eval_set = "data/Eval-sets/eval-set-bashapes.pkl"

elif DATASET == "treecycles":
    path_log = "outputs/treecycles/1653492558/log.txt"
    path_cfs = "results/syn4/SGD/syn4_cf_examples_lr0.1_beta0.5_mom0.0_epochs500"
    path_predictions = "outputs/treecycles/1653492558/predictions.pkl"
    path_eval_set = "data/Eval-sets/eval-set-treecycles.pkl"

elif DATASET == "treegrids":
    path_log = "outputs/treegrids/1653500919/log.txt"
    path_cfs = "results/syn5/SGD/syn5_cf_examples_lr0.1_beta0.5_mom0.0_epochs500"
    path_predictions = "outputs/treegrids/1653500919/predictions.pkl"
    path_eval_set = "data/Eval-sets/eval-set-treegrids.pkl"

else:
    print("Invalid dataset!")
    exit(1)
with open(path_log, "r") as file:
    log = file.readlines()

with open(path_cfs, "rb") as file:
    cfs = pickle.load(file)

with open(path_predictions, "rb") as file:
    predictions = pickle.load(file)

with open(path_eval_set, "rb") as file:
    eval_set = pickle.load(file)


## Constants
if DATASET == "bashapes":
    NUMBER_OF_LABELS = 4
else:
    NUMBER_OF_LABELS = 2
PREDICTIONS = {node:int(prediction) for node, prediction in enumerate(predictions)}
NODES_PER_PREDICTED_LABEL = defaultdict(int)
for node in PREDICTIONS:
    label = PREDICTIONS[node]
    NODES_PER_PREDICTED_LABEL[f"label-{label}"] += 1

PREDICTIONS_EVAL_SET = {node:label for node, label in PREDICTIONS.items() if node in eval_set}

NODES_PER_PREDICTED_LABEL_IN_EVAL_SET = defaultdict(int)
for node in PREDICTIONS_EVAL_SET:
    label = PREDICTIONS_EVAL_SET[node]
    NODES_PER_PREDICTED_LABEL_IN_EVAL_SET[f"label-{label}"] += 1


## Per-label explanation size
# have a dictionary for each label
per_label_explanation_size = defaultdict(list)
nodes_per_prediction = defaultdict(int)

# iterate over the cfs
for cf in cfs:
    # if: cf is []: cf wasn't found, hence skip to next iteration
    if cf == []:
        continue
    # get cf[-1][5] initial prediction over that node in the subgraph
    # We use [-1] because cfgnnexplanier stores information about the cfs in a list
    # (one entry per counterfactual edge set).
    # Each cf set is of better quality than the previous one.
    original_prediction = cf[-1][5]
    # just get cfs[-1][11] (which is the #edge-deletions)
    perturbations = cf[-1][11]
    # store this against the corresponding label in the dictionry
    per_label_explanation_size[f"label-{int(original_prediction)}"].append(int(perturbations))

for label in per_label_explanation_size:
    nodes_per_prediction[label] = len(per_label_explanation_size[label])

for label in range(NUMBER_OF_LABELS):
    # if there was no node in the eval-set with that label
    if len(per_label_explanation_size[f"label-{int(label)}"]) == 0:
        mean, std = None, None
    else:
        mean = np.mean(per_label_explanation_size[f"label-{int(label)}"])
        std = np.std(per_label_explanation_size[f"label-{int(label)}"])
    per_label_explanation_size[f"label-{int(label)}"] = [mean, std]

print("\n====================")
print("Per-label Explanation size:")
for key, value in per_label_explanation_size.items(): # format: label: (mean, std)
    print(f"{key}: {value[0]:.2f} +- {value[1]:.2f}")
print()
print(f"Nodes per predicted label in the eval-set:\n{NODES_PER_PREDICTED_LABEL_IN_EVAL_SET}\n")
print(f"Nodes per post-perturbation-prediction in the eval-set:\n{nodes_per_prediction}\n")


## Explanation size
explanation_size = list()
missed = 0
# iterate over the cfs
for cf in cfs:
    # if: cf is []: cf wasn't found, hence skip
    if cf == []:
        missed += 1
        continue
    # else: just get cfs[-1][11] (which is the #edge-deletions) in a list
    # [-1] because cfgnnexplanier stores information about the cfs in a list
    # (one entry per counterfactual edge set).
    # Each cf set is of better quality than the previous one.
    explanation_size.append(int(cf[-1][11]))
# take mean and std
explanation_size = [np.mean(explanation_size), np.std(explanation_size)]

print("\n====================")
print("Explanation_size:")
print(f"{explanation_size[0]:.2f} +- {explanation_size[1]:.2f}")
print()
print(f"#Nodes in the eval set: {len(eval_set)}")
print(f"#Nodes for which cf wasn't found: {missed}")
print(f"Hence, #nodes over which size was calculated: {len(eval_set) - missed}")


## Per-label Fidelity
nodes_for_which_cf_was_found = [cf[-1][0] for cf in cfs if cf != []]
per_label_misses = defaultdict(int)

# iterate over cfs
for node in eval_set:
    # get prediction
    label = PREDICTIONS[node]
    # check if cf was found
    if node not in nodes_for_which_cf_was_found:
        per_label_misses[f"label-{label}"] += 1

per_label_fidelity = defaultdict(int)
for label in per_label_misses:    
    per_label_fidelity[label] = per_label_misses[label]/NODES_PER_PREDICTED_LABEL_IN_EVAL_SET[label]
print("\n====================")
print("Per label fidelity:")
print(per_label_fidelity)


## Fidelity
print("\n====================")
print("Fidelity:")
fidelity = 1 - len(nodes_for_which_cf_was_found)/len(eval_set)
print(f"{fidelity:.2f}")
