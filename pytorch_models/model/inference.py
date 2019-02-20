from __future__ import absolute_import
import os
import torch
import logging
from collections import OrderedDict
from typing import Optional, List, Dict, Any

from allennlp.common.tqdm import Tqdm
from allennlp.common.params import Params
from allennlp.data.fields \
    import TextField, MultiLabelField
from allennlp.nn import util
from allennlp.data.instance import Instance
from allennlp.common.checks import ConfigurationError
from allennlp.data.tokenizers import Token
from allennlp.data import Vocabulary
from allennlp.data.instance import Instance
from allennlp.data.iterators import BasicIterator
from allennlp.data.iterators import DataIterator
from allennlp.data.token_indexers \
    import TokenIndexer, SingleIdTokenIndexer, TokenCharactersIndexer

import pytorch_models.model as Models
import pytorch_models.reader as Readers
from pytorch_models.commons.utils import read_from_config_file

logger = logging.getLogger(__name__)


class BaseModelRunner(object):
    """An Abstract class, providing a common
    way of interfacing with the Classifiers.

    Implements commonly used functionalities, like
    loading models and such.
    """

    def __init__(
            self, config_file: str, model_path: str,
            vocab_dir: str, base_embed_dir: Optional[str] = None):
        self._config = read_from_config_file(config_file)
        # Load Vocab
        logger.info("Loading Vocab")
        self._vocab = Vocabulary.from_files(vocab_dir)
        logger.info("Vocab Loaded")

        # Load Reader
        dataset_reader_params = self._config.pop("dataset_reader")
        reader_type = dataset_reader_params.pop("type", None)
        assert reader_type is not None and hasattr(Readers, reader_type),\
            f"Cannot find reader {reader_type}"
        reader = getattr(Readers, reader_type).from_params(
            dataset_reader_params)
        self._reader = reader
        # Load Model
        model_params = self._config.pop("model")
        model_type = model_params.pop("type")

        if base_embed_dir is not None:
            # This hack is necessary to ensure we load ELMO embeddings
            # correctly. Whoever discovers this later,
            # this is ugly. I apologise
            text_field_params = model_params.get("text_field_embedder", None)
            if text_field_params is not None:
                elmo_params = text_field_params.get("elmo", None)
                if elmo_params is not None:
                    options_file_path = elmo_params.get("options_file")
                    weight_file_path = elmo_params.get("weight_file")
                    _, options_file = os.path.split(options_file_path)
                    _, weight_file = os.path.split(weight_file_path)
                    complete_options_file = os.path.join(
                        base_embed_dir, options_file)
                    complete_weight_file = os.path.join(
                        base_embed_dir, weight_file)
                    elmo_params["options_file"] = complete_options_file
                    elmo_params["weight_file"] = complete_weight_file

        # clearing out the pretrained embedding file
        text_field_embedder_params = model_params.get("text_field_embedder", None)
        if text_field_embedder_params is not None:
            tokens = text_field_embedder_params.get("tokens", None)
            if tokens is not None:
                tokens.pop("pretrained_file", None)
        assert model_type is not None and hasattr(Models, model_type),\
            f"Cannot find reader {model_type}"
        self._model = getattr(Models, model_type).from_params(
            vocab=self._vocab,
            params=model_params
        )
        logger.info("Loading Model")
        model_state = torch.load(model_path,
                                 map_location=util.device_mapping(-1))
        self._model.load_state_dict(model_state)
        logger.info("Model Loaded")

        trainer_params = self._config.pop("trainer", None)
        self._cuda_device = -1
        if trainer_params is not None and torch.cuda.is_available():
            self._cuda_device = trainer_params.pop_int("cuda_device", -1)
        if self._cuda_device != -1:
            self._model.cuda(self._cuda_device)
        # This class is specifically for evaluation
        self._model.eval()

    def get_reader(self):
        return self._reader

    def get_text_from_textfield(self, text_field_instance: TextField) -> List[str]:
        return [x.text for x in text_field_instance.tokens]

    def process_instances(self, instances: List[Instance]):
        iterator_type = BasicIterator(batch_size=32)
        iterator_type.index_with(self._vocab)
        num_batches = iterator_type.get_num_batches(instances)
        iterator = iterator_type(
            instances,
            num_epochs=1,
            shuffle=False,
            cuda_device=self._cuda_device,
            for_training=False
        )
        inference_generator_tqdm = Tqdm.tqdm(iterator, total=num_batches)
        predictions = []
        index = 0
        correct_count = 0
        gold_present = False

        for batch in inference_generator_tqdm:
            output_dict = self._model.decode(self._model(**batch))
            for ix in range(len(output_dict["label"])):
                prediction = OrderedDict()
                prediction["sent_1"] = self.get_text_from_textfield(
                    instances[index].fields["premise"]
                )
                prediction["sent_2"] = self.get_text_from_textfield(
                    instances[index].fields["hypothesis"]
                )
                gold_label = instances[index].fields["label"].label \
                    if "label" in instances[index].fields \
                    else ""
                prediction["gold_label"] = gold_label
                prediction["pred_label"] = output_dict["label"][ix]
                if gold_label != "" and prediction["gold_label"] == prediction["pred_label"]:
                    correct_count += 1.
                prediction["probs"] = output_dict["probs"][ix].tolist()
                prediction["confidence"] = max(prediction["probs"])
                predictions.append(prediction)
                if gold_label != "":
                    gold_present = True
                index += 1
        if gold_present:
            print("Correct: ", correct_count / index * 100)
        return predictions

    def generate_preds_from_file(self, filename: str) -> List[Dict[str, Any]]:
        instances = self._reader.read(filename)
        return self.process_instances(instances)

    def compare_sentences(self, sent_1: str, sent_2: str, label: str = None) -> List[Dict[str, Any]]:
        instance = self._reader.text_to_instance(premise=sent_1, hypothesis=sent_2, label=label)
        return self.process_instances([instance])

    @classmethod
    def load_from_dir(cls, base_dir, base_embed_dir=None):
        """Instantiates a ModelRunner from an experiment directory
        """
        config_file = os.path.join(base_dir, "config.json")
        assert os.path.exists(config_file),\
            f"Cannot find config file in {base_dir}"
        vocab_dir = os.path.join(base_dir, "vocab")
        assert os.path.exists(vocab_dir),\
            f"Cannot find vocab dir in {base_dir}"
        model_path = os.path.join(base_dir, "models", "best.th")
        assert os.path.exists(model_path),\
            f"Cannot find Best model in {base_dir}"
        return cls(
            config_file=config_file,
            vocab_dir=vocab_dir,
            model_path=model_path,
            base_embed_dir=base_embed_dir
        )


if __name__ == "__main__":
    model_dir = "/home/scratch/bpatra/paraphrase-detection/Pytorch-Experiments/run-0"
    runner = BaseModelRunner.load_from_dir(model_dir)
    # TEST_PATH = "/home/scratch/bpatra/paraphrase-detection/Quora/test.tsv"
    # runner.generate_preds_from_file(TEST_PATH)
    examples = [
        ("let's meet next week", "next week works for me"),
        ("let's meet next week", "next week won't be possible"),
        ("today is fine", "today is fine"),
        ("today is fine", "tomorrow is fine")
    ]
    # A little more efficient comparisons
    instance_list = []
    reader = runner.get_reader()
    for example in examples:
        premise, hypothesis, label = None, None, None
        if len(example) == 2:
            premise = example[0]
            hypothesis = example[1]
        elif len(example) == 3:
            premise, hypothesis, label = example
        else:
            raise RuntimeError("Do not understand format")
        instance = reader.text_to_instance(
            premise=premise, hypothesis=hypothesis,
            label=label
        )
        instance_list.append(instance)

    predictions = runner.process_instances(instance_list)
    for pred in predictions:
        sent_1 = pred["sent_1"]
        sent_2 = pred["sent_2"]
        pred_label = pred["pred_label"]
        confidence = pred["confidence"]
        print(sent_1, sent_2, pred_label, confidence)
