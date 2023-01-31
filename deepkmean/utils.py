
from torch.nn import Module

from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.metrics.cluster import adjusted_rand_score as ari 
from sklearn.metrics.cluster import contingency_matrix as cm

import numpy as np
import torch

import json
import os

# read json config file
with open('config.json') as f:
    config = json.load(f)

class KMeansClusteringLoss(Module):
    def __init__(self, n_clusters):
        super(KMeansClusteringLoss, self).__init__()
        self.n_clusters = n_clusters
    
    def forward(self,x, cluster_assignments, cluster_centers):
        cluster_assignments = torch.from_numpy(cluster_assignments)
        
        loss = 0.0
        for i in range(self.n_clusters):
            cluster_indices = np.where(cluster_assignments == i)
            cluster_data = x[cluster_indices]
            center = cluster_centers[i]
            diff = cluster_data.detach().numpy() - center
            loss += np.sum(diff * diff)
        return torch.tensor(loss)

def algo_evaluation(trueY,predY):
    """Fonction afin d'afficher les scores NMI Et ARI
  args:
    - trueY: true labeling
    - predY : predicted Labeling

  """
    def purity_score(y_true, y_pred):
        # compute contingency matrix (also called confusion matrix)
        contingency_matrix = cm(y_true, y_pred)
        # return purity
        return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 

    nmi_res = nmi(trueY,predY)
    ari_res = ari(trueY,predY)
    pur_res = purity_score(trueY,predY)
    print(f'NMI score  :{nmi_res} ')
    print(f'ARI score  :{ari_res} ')
    print(f'Purity score : {pur_res}')

