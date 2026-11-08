import torch
import torch.nn as nn

torch.backends.mha.set_fastpath_enabled(False)

class FTTransformer(nn.Module):
    def __init__(self, d_model, n_transformer, dropout = 0.1, n_feature = 21, n_class = 3, n_head = 8):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model atau n_head tidak habis dibagi")

        bound = 1 / d_model ** 0.5

        self.weight = nn.Parameter(torch.empty(n_feature, d_model).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(n_feature, d_model).uniform_(-bound, bound))
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model).uniform_(-bound, bound))
        transformer_encoder = nn.TransformerEncoderLayer(
            d_model = d_model,
            nhead = n_head,
            dim_feedforward = d_model * 4,
            dropout = dropout,
            activation = "relu",
            batch_first = True,
            norm_first = True,
        )

        self.transformer = nn.TransformerEncoder(
            transformer_encoder,
            num_layers = n_transformer
        )

        self.transformer.layers[0].norm1 = nn.Identity()
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_class)

    def forward(self, x):
        x = x.unsqueeze(-1)

        x = x * self.weight.unsqueeze(0)
        x = x + self.bias.unsqueeze(0)

        cls = self.cls_token.expand(
            x.size(0), -1, -1
        )
        x = torch.cat([cls, x], dim=1)

        x = self.transformer(x)

        x = self.norm(x[:, 0, :])

        return self.head(torch.relu(x))
    