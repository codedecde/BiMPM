"""
BiMPM (Bilateral Multi-Perspective Matching) model implementation.
"""

from typing import Dict, Optional, List, Any

from overrides import overrides
import torch
import numpy

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.common.params import Params

from pytorch_models.model.bimpm_matching import BiMpmMatching
from pytorch_models.commons.utils import to_numpy


@Model.register("bertclassifier")
class BERTClassifier(Model):
    """
    This ``Model`` implements BiMPM model described in `Bilateral Multi-Perspective Matching
    for Natural Language Sentences <https://arxiv.org/abs/1702.03814>`_ by Zhiguo Wang et al., 2017.
    Also please refer to the `TensorFlow implementation <https://github.com/zhiguowang/BiMPM/>`_ and
    `PyTorch implementation <https://github.com/galsang/BIMPM-pytorch>`_.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``premise`` and ``hypothesis`` ``TextFields`` we get as input to the
        model.
    aggregator : ``Seq2VecEncoder``
        Aggregator of all BiMPM matching vectors
    classifier_feedforward : ``FeedForward``
        Fully connected layers for classification.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        If provided, will be used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 classifier_feedforward: FeedForward,
                 dropout : Optional[torch.nn.Dropout] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(BERTClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.dropout = dropout
        self.classifier_feedforward = classifier_feedforward
        self.metrics = {"accuracy": CategoricalAccuracy()}
        self.loss = torch.nn.CrossEntropyLoss()
        initializer(self)


    @overrides
    def forward(self,  # type: ignore
                input: Dict[str, torch.Tensor],
                label: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None  # pylint:disable=unused-argument
               ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """

        Parameters
        ----------
        premise : Dict[str, torch.LongTensor]
            The premise from a ``TextField``
        hypothesis : Dict[str, torch.LongTensor]
            The hypothesis from a ``TextField``
        label : torch.LongTensor, optional (default = None)
            The label for the pair of the premise and the hypothesis
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            Additional information about the pair
        Returns
        -------
        An output dictionary consisting of:

        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing unnormalised log
            probabilities of the entailment label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """

        pooled_output = self.text_field_embedder(input)
        if self.dropout is not None:
            pooled_output = self.dropout(pooled_output)
        # the final forward layer
        logits = self.classifier_feedforward(pooled_output)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {'logits': logits, "probs": probs}
        if label is not None:
            label = label.view(-1)
            loss = self.loss(logits, label)
            for metric in self.metrics.values():
                metric(logits, label)
            output_dict["loss"] = loss
        return output_dict

    @overrides
    def decode(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Converts indices to string labels, and adds a ``"label"``
        key to the result.
        """
        return_dict = {}
        for key in output_dict:
            return_dict[key] = to_numpy(
                output_dict[key], output_dict[key].is_cuda)
        argmax_indices = numpy.argmax(return_dict["probs"], axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        return_dict['label'] = labels
        return return_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'BERTClassifier':
        text_field_embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(
            vocab=vocab, params=text_field_embedder_params
        )
        dropout = params.pop("dropout", None)
        if dropout is not None:
            pval = dropout.pop("value")
            dropout = torch.nn.Dropout(pval)
        classifier_feedforward = FeedForward.from_params(params.pop("classifier_feedforward"))
        regularizer = RegularizerApplicator.from_params(params.pop("regularizer", None))
        return cls(
            vocab=vocab,
            text_field_embedder=text_field_embedder,
            dropout=dropout,
            classifier_feedforward=classifier_feedforward,
            regularizer=regularizer
        )