from torch import nn
from .utils import config

class autoencoder(nn.Module):
    def __init__(self):
        self.hidden_size = config['model']['hidden_size']
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(True),
            nn.Linear(128,self.hidden_size))
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(True), nn.Linear(128, 100),
            nn.Tanh())

    def forward(self, x):
        h_x = self.encoder(x)
        x = self.decoder(h_x)
        return x,h_x