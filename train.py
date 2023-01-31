from torch.utils.data import DataLoader
from torch import nn
from sklearn.cluster import KMeans
import warnings
import torch

from data.jose import Jose
from deepkmean.dkm import autoencoder
from deepkmean.utils import KMeansClusteringLoss
from deepkmean.utils import algo_evaluation
from deepkmean.utils import config


dataset=Jose('data/jose_d.txt',target_file='data/label.txt',sep=' ',header=None)
dataloader=DataLoader(dataset,batch_size=config['training']['batch_size'],shuffle=True)

alpha = config['training']['alpha']
ncluster = config['training']['num_clusters']

model = autoencoder()
criterion = nn.MSELoss()
kmeans_loss = KMeansClusteringLoss(ncluster)
optimizer = torch.optim.Adam(
    model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])

print(f'[alpha={alpha}]')
for epoch in range(config['training']['epochs']):
    
    for data in dataloader:
        x_, y_ = data
        # forward
        output,hidden_state = model(x_)
        kmeans = KMeans(n_init=10,n_clusters=ncluster, random_state=0)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kmeans.fit(hidden_state.data.numpy())

        kloss = kmeans_loss(hidden_state,kmeans.labels_,kmeans.cluster_centers_) #kmean loss
        loss = criterion(output, x_) + alpha*kloss # over_all loss
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, config['training']['epochs'], loss.item()))

# Evaluation
output, hidden_state = model(dataset.x_train)
dkmeans = KMeans(n_init=10,n_clusters=ncluster)
dkmeans.fit(hidden_state.data.numpy())
algo_evaluation(dataset.y_train,dkmeans.labels_)
