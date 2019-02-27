"""
A ``TokenEmbedder`` which uses one of the BERT models
(https://github.com/google-research/bert)
to produce embeddings.

At its core it uses Hugging Face's PyTorch implementation
(https://github.com/huggingface/pytorch-pretrained-BERT),
so thanks to them!
"""
import logging

import torch

from pytorch_pretrained_bert.modeling import BertModel

from allennlp.common import Params
from allennlp.modules.scalar_mix import ScalarMix
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.nn import util
from allennlp.data import Vocabulary

logger = logging.getLogger(__name__)


class BertEmbedder(TokenEmbedder):
    """
    A ``TokenEmbedder`` that produces BERT embeddings for your tokens.
    Should be paired with a ``BertIndexer``, which produces wordpiece ids.

    Most likely you probably want to use ``PretrainedBertEmbedder``
    for one of the named pretrained models, not this base class.

    Parameters
    ----------
    bert_model: ``BertModel``
        The BERT model being wrapped.
    top_layer_only: ``bool``, optional (default = ``False``)
        If ``True``, then only return the top layer instead of apply the scalar mix.
    """
    def __init__(
        self, bert_model: BertModel,
        top_layer_only: bool = False, for_classification: bool = False) -> None:
        super().__init__()
        self.bert_model = bert_model
        self.output_dim = bert_model.config.hidden_size
        self.for_classification = for_classification
        if not self.for_classification:
            if not top_layer_only:
                self._scalar_mix = ScalarMix(bert_model.config.num_hidden_layers,
                                             do_layer_norm=False)
            else:
                self._scalar_mix = None

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(self,
                input_ids: torch.LongTensor,
                offsets: torch.LongTensor = None,
                token_type_ids: torch.LongTensor = None) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids : ``torch.LongTensor``
            The (batch_size, ..., max_sequence_length) tensor of wordpiece ids.
        offsets : ``torch.LongTensor``, optional
            The BERT embeddings are one per wordpiece. However it's possible/likely
            you might want one per original token. In that case, ``offsets``
            represents the indices of the desired wordpiece for each original token.
            Depending on how your token indexer is configured, this could be the
            position of the last wordpiece for each token, or it could be the position
            of the first wordpiece for each token.

            For example, if you had the sentence "Definitely not", and if the corresponding
            wordpieces were ["Def", "##in", "##ite", "##ly", "not"], then the input_ids
            would be 5 wordpiece ids, and the "last wordpiece" offsets would be [3, 4].
            If offsets are provided, the returned tensor will contain only the wordpiece
            embeddings at those positions, and (in particular) will contain one embedding
            per token. If offsets are not provided, the entire tensor of wordpiece embeddings
            will be returned.
        token_type_ids : ``torch.LongTensor``, optional
            If an input consists of two sentences (as in the BERT paper),
            tokens from the first sentence should have type 0 and tokens from
            the second sentence should have type 1.  If you don't provide this
            (the default BertIndexer doesn't) then it's assumed to be all 0s.
        """
        # pylint: disable=arguments-differ
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        input_mask = (input_ids != 0).long()

        # input_ids may have extra dimensions, so we reshape down to 2-d
        # before calling the BERT model and then reshape back at the end.
        
        all_encoder_layers, classification_out = self.bert_model(
            input_ids=util.combine_initial_dims(input_ids),
            token_type_ids=util.combine_initial_dims(token_type_ids),
            attention_mask=util.combine_initial_dims(input_mask)
        )
        if self.for_classification:
            return classification_out
        else:
            if self._scalar_mix is not None:
                mix = self._scalar_mix(all_encoder_layers, input_mask)
            else:
                mix = all_encoder_layers[-1]

            # At this point, mix is (batch_size * d1 * ... * dn, sequence_length, embedding_dim)

            if offsets is None:
                # Resize to (batch_size, d1, ..., dn, sequence_length, embedding_dim)
                return util.uncombine_initial_dims(mix, input_ids.size())
            else:
                # offsets is (batch_size, d1, ..., dn, orig_sequence_length)
                offsets2d = util.combine_initial_dims(offsets)
                # now offsets is (batch_size * d1 * ... * dn, orig_sequence_length)
                range_vector = util.get_range_vector(offsets2d.size(0),
                                                     device=util.get_device_of(mix)).unsqueeze(1)
                # selected embeddings is also (batch_size * d1 * ... * dn, orig_sequence_length)
                selected_embeddings = mix[range_vector, offsets2d]

                return util.uncombine_initial_dims(selected_embeddings, offsets.size())

        @overrides
        def state_dict(self, destination=None, prefix='', keep_vars=False):
            r"""Returns a dictionary containing a whole state of the module.
            Both parameters and persistent buffers (e.g. running averages) are
            included. Keys are corresponding parameter and buffer names.
            Returns:
                dict:
                    a dictionary containing a whole state of the module
            Example::
                >>> module.state_dict().keys()
                ['bias', 'weight']

            The reason we want to separate this from the usual pytorch one, is
            for some reason, the HuggingFace LayerNorm uses weight and bias instead of
            gamma and beta.
            """
            if destination is None:
                destination = OrderedDict()
                destination._metadata = OrderedDict()
            destination._metadata[prefix[:-1]] = local_metadata = dict(version=self._version)
            for name, param in self._parameters.items():
                if param is not None:
                    destination[prefix + name] = param if keep_vars else param.data
            for name, buf in self._buffers.items():
                if buf is not None:
                    destination[prefix + name] = buf if keep_vars else buf.data
            for name, module in self._modules.items():
                if module is not None:
                    module.state_dict(destination, prefix + name + '.', keep_vars=keep_vars)
            for hook in self._state_dict_hooks.values():
                hook_result = hook(self, destination, prefix, local_metadata)
                if hook_result is not None:
                    destination = hook_result
            # This part is different
            new_keys = []
            old_keys = []
            for key in destination.keys():
                new_key = None
                if "gamma" in key:
                    new_key = key.replace("gamma", "weight")
                if "beta" in key:
                    new_key = key.replace("beta", "bias")
                if new_key:
                    old_keys.append(key)
                    new_keys.append(new_key)
            for old_key, new_key in zip(old_keys, new_keys):
                destination[new_key] = destination.pop(old_key)
            return destination


@TokenEmbedder.register("bert-pretrained")
class PretrainedBertEmbedder(BertEmbedder):
    # pylint: disable=line-too-long
    """
    Parameters
    ----------
    pretrained_model: ``str``
        Either the name of the pretrained model to use (e.g. 'bert-base-uncased'),
        or the path to the .tar.gz file with the model weights.

        If the name is a key in the list of pretrained models at
        https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py#L41
        the corresponding path will be used; otherwise it will be interpreted as a path or URL.
    requires_grad : ``bool``, optional (default = False)
        If True, compute gradient of BERT parameters for fine tuning.
    top_layer_only: ``bool``, optional (default = ``False``)
        If ``True``, then only return the top layer instead of apply the scalar mix.
    """
    def __init__(
        self, pretrained_model: str,
        requires_grad: bool = False,
        top_layer_only: bool = False,
        for_classification: bool = False) -> None:
        model = BertModel.from_pretrained(pretrained_model)

        for param in model.parameters():
            param.requires_grad = requires_grad

        super().__init__(
            bert_model=model,
            top_layer_only=top_layer_only,
            for_classification=for_classification
        )

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params)->'PretrainedBertEmbedder':
        pretrained_model = params.pop("pretrained_model")
        requires_grad = params.pop("requires_grad", False)
        top_layer_only = params.pop("top_layer_only", False)
        for_classification = params.pop("for_classification", False)
        params.assert_empty(cls.__name__)
        return cls(
            pretrained_model=pretrained_model,
            requires_grad=requires_grad,
            top_layer_only=top_layer_only,
            for_classification=for_classification
        )
