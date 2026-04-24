import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=100, features=64):
        super().__init__()
        self.net = nn.Sequential(
            self._block(z_dim, features * 4, 4, 1, 0),
            self._block(features * 4, features * 2, 4, 2, 1),
            self._block(features * 2, features, 4, 2, 1),
            nn.ConvTranspose2d(features, 3, 4, 2, 1),
            nn.Tanh()
        )

    def _block(self, in_c, out_c, k, s, p):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, features=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, features, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            self._block(features, features * 2, 4, 2, 1),
            self._block(features * 2, features * 4, 4, 2, 1),
            nn.Conv2d(features * 4, 1, 4, 1, 0)
        )

    def _block(self, in_c, out_c, k, s, p):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.net(x).view(-1)