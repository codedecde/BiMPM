from typing import Dict, List
import logging
import csv

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field
from allennlp.data.instance import Instance
from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import BertBasicWordSplitter
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.common.params import Params
from allennlp.common.tqdm import Tqdm

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

SEP_TOKEN = Token("[SEP]")


@DatasetReader.register("bert_paraphrase")
class BERTParaphraseReader(DatasetReader):
    """
    """

    def __init__(
        self,
        lazy: bool = False,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None
    ) -> None:
        super(BERTParaphraseReader, self).__init__(lazy)
        # By default we use the BERT Wordsplitter
        self._tokenizer = tokenizer or WordTokenizer(BertBasicWordSplitter())
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str) -> List[Instance]:
        logger.info("Reading instances from lines in file at: %s", file_path)
        instances = []
        with open(cached_path(file_path), "r") as data_file:
            # tsv_in = csv.reader(data_file, delimiter='\t')
            for line in data_file:
                row = line.split(u"\t")
                if len(row) == 4:
                    instances.append(self.text_to_instance(
                        premise=row[1], hypothesis=row[2], label=row[0]))
        return instances

    @overrides
    def text_to_instance(
        self, premise: str,
        hypothesis: str,
        label: str = None
    ) -> Instance:
        fields: Dict[str, Field] = {}
        tokenized_premise = self._tokenizer.tokenize(premise)
        tokenized_hypothesis = self._tokenizer.tokenize(hypothesis)
        # BERT expects both sentences concatenated by a "[SEP]"
        input_sentence = tokenized_premise + \
            [SEP_TOKEN] + tokenized_hypothesis
        fields["input"] = TextField(input_sentence, self._token_indexers)
        if label is not None:
            fields["label"] = LabelField(label)

        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'BERTParaphraseReader':
        token_indexers = TokenIndexer.dict_from_params(
            params.pop("token_indexers", {})
        )
        tokenizer = Tokenizer.from_params(
            params.pop("tokenizer", {})
        )
        lazy = params.pop("lazy", False)
        return cls(
            lazy=lazy,
            tokenizer=tokenizer,
            token_indexers=token_indexers
        )
