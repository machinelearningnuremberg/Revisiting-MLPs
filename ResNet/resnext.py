# %%
import math
import typing as ty
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import zero
from torch import Tensor

import lib


# %%
class ResNext(nn.Module):
    def __init__(
        self,
        *,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_embedding: int,
        d: int,
        d_hidden_factor: float,
        n_layers: int,
        activation: str,
        normalization: str,
        hidden_dropout: float,
        residual_dropout: float,
        d_out: int,
        cardinality: int,
    ) -> None:
        super().__init__()

        def make_normalization():
            return {'batchnorm': nn.BatchNorm1d, 'layernorm': nn.LayerNorm}[
                normalization
            ](d)

        self.main_activation = lib.get_activation_fn(activation)
        self.last_activation = lib.get_nonglu_activation_fn(activation)
        self.residual_dropout = residual_dropout
        self.hidden_dropout = hidden_dropout
        self.cardinality = cardinality

        d_in = d_numerical
        d_hidden = int(d * d_hidden_factor)
        d_hidden_per_path = int(d_hidden / self.cardinality)

        if categories is not None:
            d_in += len(categories) * d_embedding
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_embedding)
            self.categories = torch.tensor(np.subtract(categories, 1).tolist())
            self.unknown_value = np.iinfo('int64').max - 3
            nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            print(f'{self.category_embeddings.weight.shape=}')

        self.first_layer = nn.Linear(d_in, d)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        'norm': make_normalization(),
                        'linear0': nn.ModuleList([nn.Linear(d, d_hidden_per_path) for _ in range(cardinality)]),
                        'linear1': nn.ModuleList([nn.Linear(d_hidden_per_path, d) for _ in range(cardinality)]),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        self.last_normalization = make_normalization()
        self.head = nn.Linear(d, d_out)

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
        x = []
        if x_num is not None:
            x.append(x_num)
        if x_cat is not None:
            x_cat = torch.where(x_cat == self.unknown_value, self.categories.to(x_cat.device), x_cat)
            x.append(
                self.category_embeddings(x_cat + self.category_offsets[None]).view(
                    x_cat.size(0), -1
                )
            )
        x = torch.cat(x, dim=-1)

        x = self.first_layer(x)
        for layer in self.layers:
            layer = ty.cast(ty.Dict[str, nn.Module], layer)
            z = x
            z = layer['norm'](z) if z.shape[0] > 1 else z
            path_outputs = []
            for i in range(self.cardinality):
                path_output = layer['linear0'][i](z)
                path_output = self.main_activation(path_output)
                if self.hidden_dropout:
                    path_output = F.dropout(path_output, p=self.hidden_dropout, training=self.training)
                path_output = layer['linear1'][i](path_output)
                if self.residual_dropout:
                    path_output = F.dropout(path_output, self.residual_dropout, self.training)
                path_outputs.append(path_output)
            z = sum(path_outputs)
            x = x + z
        x = self.last_normalization(x) if x.shape[0] > 1 else x
        x = self.last_activation(x)
        x = self.head(x)
        x = x.squeeze(-1)
        return x
