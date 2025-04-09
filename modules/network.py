import torch.nn as nn
import torch
from torch.nn.functional import normalize
from modules.decoder import Decoder

class Network(nn.Module):
    def __init__(self, resnet):
        super(Network, self).__init__()
        self.resnet = resnet
        self.feature_dim = resnet.fc.out_features
        self.predict_projector = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 10),
            nn.LeakyReLU(),
            nn.Linear(self.feature_dim // 10, self.feature_dim // 5),
            nn.LeakyReLU(),
            nn.Linear(self.feature_dim // 5, 1),

        )
        self.decoder = Decoder(in_features=self.feature_dim, out_channels=3)

    def forward(self, x):
        x = self.resnet(x)
        pre = self.predict_projector(x)
        recon = self.decoder(x)
        return pre, recon

