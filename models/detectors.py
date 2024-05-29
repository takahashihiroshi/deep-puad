from typing import Optional

import torch
from torch import nn


class Detector(nn.Module):
    def estimate(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Autoencoder(Detector):
    def __init__(self, n_in: int, n_latent: int, n_h: int):
        super().__init__()

        self.n_in = n_in
        self.n_latent = n_latent
        self.n_h = n_h

        self.encoder = nn.Sequential(
            nn.Linear(n_in, n_h),
            nn.ReLU(),
            nn.Linear(n_h, n_h),
            nn.ReLU(),
            nn.Linear(n_h, n_h),
            nn.ReLU(),
            nn.Linear(n_h, n_latent),
        )

        self.decoder = nn.Sequential(
            nn.Linear(n_latent, n_h),
            nn.ReLU(),
            nn.Linear(n_h, n_h),
            nn.ReLU(),
            nn.Linear(n_h, n_h),
            nn.ReLU(),
            nn.Linear(n_h, n_in),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return torch.norm(x - self.decoder(z), dim=1)  # type: ignore

    def estimate(self, x: torch.Tensor) -> torch.Tensor:
        return 1 - torch.exp(-self.forward(x))  # type: ignore


class DAE(Autoencoder):
    def __init__(self, n_in: int, n_latent: int, n_h: int, epsilon: float):
        super().__init__(n_in=n_in, n_latent=n_latent, n_h=n_h)
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x + self.epsilon * torch.randn_like(x))
        return torch.norm(x - self.decoder(z), dim=1)  # type: ignore

    def estimate(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        loss = torch.norm(x - self.decoder(z), dim=1)
        return 1 - torch.exp(-loss)  # type: ignore


class DeepSVDD(Detector):
    def __init__(self, n_in: int, n_latent: int, n_h: int):
        super().__init__()

        self.n_in = n_in
        self.n_latent = n_latent
        self.n_h = n_h

        self.encoder = nn.Sequential(
            nn.Linear(n_in, n_h, bias=False),
            nn.ReLU(),
            nn.Linear(n_h, n_h, bias=False),
            nn.ReLU(),
            nn.Linear(n_h, n_h, bias=False),
            nn.ReLU(),
            nn.Linear(n_h, n_latent, bias=False),
        )

        self.decoder = nn.Sequential(
            nn.Linear(n_latent, n_h, bias=False),
            nn.ReLU(),
            nn.Linear(n_h, n_h, bias=False),
            nn.ReLU(),
            nn.Linear(n_h, n_h, bias=False),
            nn.ReLU(),
            nn.Linear(n_h, n_in, bias=False),
        )

        self.c: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.c is None:
            z = self.encoder(x)
            return torch.norm(x - self.decoder(z), dim=1)  # type: ignore
        else:
            z = self.encoder(x)
            return torch.norm(z - self.c, dim=1) ** 2  # type: ignore

    def estimate(self, x: torch.Tensor) -> torch.Tensor:
        return 1 - torch.exp(-self.forward(x))  # type: ignore


class MLP(Detector):
    def __init__(self, n_in: int, n_h: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(n_in, n_h),
            nn.ReLU(),
            nn.Linear(n_h, n_h),
            nn.ReLU(),
            nn.Linear(n_h, n_h),
            nn.ReLU(),
            nn.Linear(n_h, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.squeeze(self.classifier(x))

    def estimate(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))
