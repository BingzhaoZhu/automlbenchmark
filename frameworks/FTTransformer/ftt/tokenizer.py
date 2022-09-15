import torch
from torch import Tensor, nn
from typing import List, Optional
from .transformer import _TokenInitialization


class FusionTokenizer(nn.Module):
    def __init__(
        self,
        cat_dims,
        n_con,
        emb_dim,
        flatten=True
    ):
        super().__init__()

        self.tokenizer_cat = CategoricalFeatureTokenizer(cat_dims, emb_dim)
        self.tokenizer_con = NumericalFeatureTokenizer(n_con, emb_dim)
        self.flatten = flatten

    def forward(self, anchor_cat, anchor_con):
        x_cat = self.tokenizer_cat(anchor_cat)
        x_con = self.tokenizer_con(anchor_con)
        anchor = torch.cat([x_cat, x_con], 1)

        return torch.flatten(anchor, start_dim=1) if self.flatten else anchor


class CategoricalFeatureTokenizer(nn.Module):
    """
    Feature tokenizer for categorical features in tabular data.
    It transforms the input categorical features to tokens (embeddings).
    The categorical features usually refers to discrete features.
    """

    def __init__(
        self,
        num_categories: List[int],
        d_token: int,
        bias: Optional[bool] = True,
        initialization: Optional[str] = "normal",
    ) -> None:
        """
        Parameters
        ----------
        num_categories:
            A list of integers. Each one is the number of categories in one categorical column.
        d_token:
            The size of one token.
        bias:
            If `True`, for each feature, an additional trainable vector will be added to the
            embedding regardless of feature value. Notablly, the bias are not shared between features.
        initialization:
            Initialization policy for parameters. Must be one of `['uniform', 'normal']`.
        References
        ----------
        1. Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko,
        "Revisiting Deep Learning Models for Tabular Data", 2021
        https://arxiv.org/pdf/2106.11959.pdf
        2. Code: https://github.com/Yura52/tabular-dl-revisiting-models
        """
        super().__init__()

        self.num_categories = num_categories
        category_offsets = torch.tensor([0] + num_categories[:-1]).cumsum(0)

        self.register_buffer("category_offsets", category_offsets, persistent=False)
        self.embeddings = nn.Embedding(sum(num_categories), d_token)
        self.bias = nn.Parameter(Tensor(len(num_categories), d_token)) if bias else None
        initialization_ = _TokenInitialization.from_str(initialization)

        for parameter in [self.embeddings.weight, self.bias]:
            if parameter is not None:
                initialization_.apply(parameter, d_token)

    @property
    def n_tokens(self) -> int:
        """The number of tokens."""
        return len(self.num_categories)

    @property
    def d_token(self) -> int:
        """The size of one token."""
        return self.embeddings.embedding_dim

    def forward(self, x: Tensor) -> Tensor:
        x = self.embeddings(x + self.category_offsets[None])

        if self.bias is not None:
            x = x + self.bias[None]

        return x


class NumericalFeatureTokenizer(nn.Module):
    """
    Numerical tokenizer for numerical features in tabular data.
    It transforms the input numerical features to tokens (embeddings).
    The numerical features usually refers to continous features.
    It consists of two steps:
        1. each feature is multiplied by a trainable vector i.e., weights,
        2. another trainable vector is added i.e., bias.
    Note that each feature has its separate pair of trainable vectors,
    i.e. the vectors are not shared between features.
    """

    def __init__(
        self,
        in_features: int,
        d_token: int,
        bias: Optional[bool] = True,
        initialization: Optional[str] = "normal",
    ):
        """
        Parameters
        ----------
        in_features:
            Dimension of input features i.e. the number of continuous (scalar) features
        d_token:
            The size of one token.
        bias:
            If `True`, for each feature, an additional trainable vector will be added to the
            embedding regardless of feature value. Notablly, the bias are not shared between features.
        initialization:
            Initialization policy for parameters. Must be one of `['uniform', 'normal']`.
        References
        ----------
        Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko,
        "Revisiting Deep Learning Models for Tabular Data", 2021
        https://arxiv.org/pdf/2106.11959.pdf
        """
        super().__init__()

        initialization_ = _TokenInitialization.from_str(initialization)
        self.weight = nn.Parameter(Tensor(in_features, d_token))
        self.bias = nn.Parameter(Tensor(in_features, d_token)) if bias else None
        for parameter in [self.weight, self.bias]:
            if parameter is not None:
                initialization_.apply(parameter, d_token)

    @property
    def n_tokens(self) -> int:
        """The number of tokens."""
        return len(self.weight)

    @property
    def d_token(self) -> int:
        """The size of one token."""
        return self.weight.shape[1]

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        x = self.weight[None] * x[..., None]
        if self.bias is not None:
            x = x + self.bias[None]

        return x