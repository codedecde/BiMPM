from overrides import overrides

import torch

from allennlp.common import Params
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder


@Seq2VecEncoder.register("bert")
class BERTEncoder(Seq2VecEncoder):
    """
    This is a simple encoder that works with the BERT model
    BERT uses the [CLS] token for classification, which
    is the first token. So this provides an easy way of
    extracting that.

    Parameters
    ----------
    embedding_dim: ``int``
        This is the input dimension to the encoder
    """
    def __init__(self, embedding_dim: int) -> None:
        super(BERTEncoder, self).__init__()
        self._embedding_dim = embedding_dim

    @overrides
    def get_input_dim(self) -> int:
        return self._embedding_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._embedding_dim

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor = None):
        return tokens[:, 0]

    @classmethod
    def from_params(cls, params: Params) -> 'BERTEncoder':
        embedding_dim = params.pop_int("embedding_dim")
        params.assert_empty(cls.__name__)
        return cls(embedding_dim=embedding_dim)
