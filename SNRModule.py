import torch
import torch.nn as nn


class SNRModulation(nn.Module):
    def __init__(self, embed_dim, num_layers=3, hidden_dim=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim if hidden_dim is not None else int(embed_dim * 1.5)

        self.feature_transforms = nn.ModuleList()
        self.snr_modulators = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.feature_transforms.append(nn.Linear(embed_dim, self.hidden_dim))
            else:
                self.feature_transforms.append(nn.Linear(self.hidden_dim, self.hidden_dim))

        for i in range(num_layers):
            self.snr_modulators.append(nn.Sequential(
                nn.Linear(1, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Sigmoid()
            ))

        self.final_transform = nn.Sequential(
            nn.Linear(self.hidden_dim, embed_dim),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x, snr):
        """
        Args:
            x: input features [B, embed_dim], or [B, T, embed_dim]
            snr: signal-to-noise ratio value
        Returns:
            modulated features [B, embed_dim]
        """
        B, T, D = x.shape
        snr_tensor = torch.full((B, T, 1), snr, dtype=x.dtype, device=x.device)

        # detach gradient
        temp = self.feature_transforms[0](x.detach())

        for i in range(self.num_layers):
            snr_mod = self.snr_modulators[i](snr_tensor)
            temp = temp * snr_mod

            if i < self.num_layers - 1:
                temp = self.feature_transforms[i + 1](temp)

        mod_weights = self.final_transform(temp)
        modulated_x = x * mod_weights

        return modulated_x
