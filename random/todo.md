# CF-GNNEXPLAINER

- Pickle files containing cfs are stored in the results folder. [AUTHOR]
- The pickled file for each dataset contains lists of the following format:

keys = [
    'node_idx',
    'new_idx',
    'cf_adj',
    'sub_adj',
    'y_pred_orig',
    'y_pred_new',
    'y_pred_new_actual',
    'sub_labels[new_idx]',
    'sub_adj',
    'loss_total',
    'loss_pred',
    'loss_graph_dist'
]

- Note that, nodes for which a cf is not found are blank lists in the pickled files.

## Script design

bash get_outputs.sh DATASET
