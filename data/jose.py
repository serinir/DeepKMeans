from torch.utils.data import Dataset, DataLoader
import torch

import pandas as pd
import numpy as np

class Jose(Dataset):
  def __init__(self,file_name,target_file=None,sep=' ',header=None):
    data=pd.read_csv(file_name,sep=sep,header=header)
    data.drop(columns=[0,101],inplace=True)
    target = pd.read_csv(target_file,header=None).values

    X = np.array(data)
    y = np.array(target,dtype=np.int32)
 
    self.x_train=torch.tensor(X,dtype=torch.float32)
    self.y_train=torch.tensor(y,dtype=torch.float32).reshape(-1)
  def __len__(self):
    return len(self.y_train)
   
  def __getitem__(self,idx):
    return self.x_train[idx],self.y_train[idx]