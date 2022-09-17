from typing import List, Optional

import torch
from torch import Tensor, nn
from .transformer import CLSToken, FT_Transformer, _TokenInitialization
from .tokenizer import FusionTokenizer


# class PreTrainHead(nn.Module):
#     """The pre-training head of the `FTTransformer`."""
#
#     def __init__(
#             self,
#             *,
#             d_in: int,
#             bias: bool,
#             d_out: int,
#     ):
#         super().__init__()
#         self.linear = nn.Linear(d_in, d_out, bias)
#         self.nn.ReLU(inplace=True)
#
#     def forward(self, x: Tensor) -> Tensor:
#         x = x[:, -1]
#         x = self.linear(x)
#         return x

class FTTransformer_(nn.Module):
    """
    FT-Transformer for numerical tabular features.
    """

    def __init__(
        self,
        prefix: str,
        num_categories: List[int],
        in_features: int,
        d_token: int,
        cls_token: Optional[bool] = True,
        out_features: Optional[int] = None,
        num_classes: Optional[int] = 0,
        token_initialization: Optional[str] = "normal",
        n_blocks: Optional[int] = 0,
        attention_n_heads: Optional[int] = 8,
        attention_initialization: Optional[str] = "kaiming",
        attention_normalization: Optional[str] = "layer_norm",
        attention_dropout: Optional[float] = 0.2,
        residual_dropout: Optional[float] = 0.0,
        ffn_activation: Optional[str] = "reglu",
        ffn_normalization: Optional[str] = "layer_norm",
        ffn_d_hidden: Optional[int] = 192,
        ffn_dropout: Optional[float] = 0.0,
        prenormalization: Optional[bool] = True,
        first_prenormalization: Optional[bool] = False,
        kv_compression_ratio: Optional[float] = None,
        kv_compression_sharing: Optional[str] = None,
        head_activation: Optional[str] = "relu",
        head_normalization: Optional[str] = "layer_norm",
        embedding_arch: Optional[List[str]] = ["linear"],
        train_status: Optional[str] = "pretrain",
    ):
        """
        Parameters
        ----------
        prefix
            The model prefix.
        in_features
            Dimension of input features.
        d_token
            The size of one token for `NumericalEmbedding`.
        cls_token
            If `True`, [cls] token will be added to the token embeddings.
        out_features
            Dimension of output features.
        num_classes
            Number of classes. 1 for a regression task.
        token_bias
            If `True`, for each feature, an additional trainable vector will be added in `_CategoricalFeatureTokenizer`
            to the embedding regardless of feature value. Notablly, the bias are not shared between features.
        token_initialization
            Initialization policy for parameters in `_CategoricalFeatureTokenizer` and `_CLSToke`.
            Must be one of `['uniform', 'normal']`.
        n_blocks
            Number of the `FT_Transformer` blocks, which should be non-negative.
        attention_n_heads
            Number of attention heads in each `FT_Transformer` block, which should be postive.
        attention_initialization
            Weights initalization scheme for Multi Headed Attention module.
        attention_dropout
            Dropout ratio for the Multi Headed Attention module.
        residual_dropout
            Dropout ratio for the linear layers in FT_Transformer block.
        ffn_activation
            Activation function type for the Feed-Forward Network module.
        ffn_normalization
            Normalization scheme of the Feed-Forward Network module.
        ffn_d_hidden
            Number of the hidden nodes of the linaer layers in the Feed-Forward Network module.
        ffn_dropout
            Dropout ratio of the hidden nodes of the linaer layers in the Feed-Forward Network module.
        prenormalization, first_prenormalization
            Prenormalization to stablize the training.
        kv_compression_ratio
            The compression ration to reduce the input sequence length.
        kv_compression_sharing
            If `true` the projections will share weights.
        head_activation
            Activation function type of the MLP layer.
        head_normalization
            Normalization scheme of the MLP layer.
        embedding_arch
            A list containing the names of embedding layers.
            Currently support:
            {'linear', 'shared_linear', 'autodis', 'positional', 'relu', 'layernorm'}
        References
        ----------
        1. Paper: Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko,
        "Revisiting Deep Learning Models for Tabular Data", 2021 https://arxiv.org/pdf/2106.11959.pdf
        2. Paper: On Embeddings for Numerical Features in Tabular Deep Learning, https://arxiv.org/abs/2203.05556
        3. Code: https://github.com/Yura52/tabular-dl-revisiting-models
        4. Code: https://github.com/Yura52/tabular-dl-num-embeddings
        """

        super().__init__()

        assert d_token > 0, "d_token must be positive"
        assert n_blocks >= 0, "n_blocks must be non-negative"
        assert attention_n_heads > 0, "attention_n_heads must be postive"
        assert token_initialization in ["uniform", "normal"], "initialization must be uniform or normal"
        assert train_status in ["pretrain", "finetune"], "training status must be pretrain or finetune"

        self.train_status = train_status
        self.prefix = prefix
        self.out_features = out_features

        self.tokenizer = FusionTokenizer(
            cat_dims=num_categories,
            n_con=in_features,
            emb_dim=d_token,
            flatten=False,
        )

        self.cls_token = (
            CLSToken(
                d_token=d_token,
                initialization=token_initialization,
            )
            if cls_token
            else nn.Identity()
        )

        if kv_compression_ratio is not None:
            if self.cls_token:
                n_tokens = self.numerical_feature_tokenizer.n_tokens + 1
            else:
                n_tokens = self.numerical_feature_tokenizer.n_tokens
        else:
            n_tokens = None

        self.transformer = FT_Transformer(
            d_token=d_token,
            n_blocks=n_blocks,
            attention_n_heads=attention_n_heads,
            attention_dropout=attention_dropout,
            attention_initialization=attention_initialization,
            attention_normalization=attention_normalization,
            ffn_d_hidden=ffn_d_hidden,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation,
            ffn_normalization=ffn_normalization,
            residual_dropout=residual_dropout,
            prenormalization=prenormalization,
            first_prenormalization=first_prenormalization,
            last_layer_query_idx=None,
            n_tokens=n_tokens,
            kv_compression_ratio=kv_compression_ratio,
            kv_compression_sharing=kv_compression_sharing,
            head_activation=head_activation,
            head_normalization=head_normalization,
            d_out=out_features,
        )

        self.prehead = FT_Transformer.Head(
            d_in=d_token,
            d_out=d_token,
            bias=True,
            activation=head_activation,
            normalization=head_normalization if prenormalization else "Identity",
        )

        self.head = FT_Transformer.Head(
            d_in=d_token,
            d_out=num_classes,
            bias=True,
            activation=head_activation,
            normalization=head_normalization if prenormalization else "Identity",
        )

        self.name_to_id = self.get_layer_ids()

    @property
    def numerical_key(self):
        return f"{self.prefix}_numerical"

    @property
    def label_key(self):
        return f"{self.prefix}_label"

    def forward(self, anchor_cat, anchor_con):
        """
        Parameters
        ----------
        batch
            A dictionary containing the input mini-batch data.
            We need to use the keys with the model prefix to index required data.
        Returns
        -------
            A dictionary with logits and features.
        """
        is_pretrai = self.train_status == "pretrain"


        features = self.tokenizer(anchor_cat, anchor_con)
        features = self.cls_token(features)
        features = self.transformer(features)
        if self.train_status == "pretrain":
            embedding = self.prehead(features)
        else:
            embedding = self.head(features)

        # positive_cat, positive_con = self.corruption(anchor_cat).type(torch.int64), self.corruption(anchor_con)
        # positive = self.tokenizer(positive_cat, positive_con)
        # positive = self.cls_token(positive)
        # positive = self.transformer(positive)
        # if self.train_status == "pretrain":
        #     embedding_pos = self.prehead(positive)
        # else:
        #     embedding_pos = self.head(positive)

        return embedding #,  embedding_pos

    def set_train_status(self, train_status):
        assert train_status in ["pretrain", "finetune"], "training status must be pretrain or finetune"
        self.train_status = train_status
        return

    def get_layer_ids(
        self,
    ):
        """
        All layers have the same id 0 since there is no pre-trained models used here.
        Returns
        -------
        A dictionary mapping the layer names (keys) to their ids (values).
        """
        name_to_id = {}
        for n, _ in self.named_parameters():
            name_to_id[n] = 0

        return name_to_id

    def get_embeddings(self, anchor_cat, anchor_con):
        features = self.tokenizer(anchor_cat, anchor_con)
        features = self.cls_token(features)
        features = self.transformer(features)
        embedding = features[:, -1]
        return embedding
