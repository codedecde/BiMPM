from __future__ import absolute_import
import argparse
import numpy as np
import torch
import os
import sys
import logging
import pdb


from allennlp.data import Vocabulary
from allennlp.training.trainer import Trainer
from allennlp.data.iterators import DataIterator

import pytorch_models.reader as Readers

# import model as Models
import pytorch_models.model as Models

from pytorch_models.commons.utils import \
    setup_output_dir, read_from_config_file



def get_arguments():
    parser = argparse.ArgumentParser(description="Paraphrase Detection")
    parser.add_argument('-cf', '--config_file', action="store",
                        dest="config_file", type=str,
                        help="path to the config file", required=True)
    parser.add_argument('-l', '--log', action="store", dest="loglevel",
                        type=str, default="INFO", help="Logging Level")
    parser.add_argument('-s', '--seed', action="store", dest="seed", type=int,
                        default=-1, help="use fixed random seed")
    args = parser.parse_args(sys.argv[1:])
    args.loglevel = args.loglevel.upper()
    return args


def main():
    """The main entry point

    This is the main entry point for training HAN SOLO models.

    Usage::

        ${PYTHONPATH} -m AttentionSegmentation/main
            --config_file ${CONFIG_FILE}

    """
    args = get_arguments()
    # Setup Experiment Directory
    config = read_from_config_file(args.config_file)
    if args.seed > 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if config.get('trainer', None) is not None and \
           config.get('trainer', None).get('cuda_device', -1) > 0:
            torch.cuda.manual_seed(args.seed)
    serial_dir, config = setup_output_dir(config, args.loglevel)
    logger = logging.getLogger(__name__)

    # Data Loaders ...
    dataset_reader_params = config.pop("dataset_reader")
    reader_type = dataset_reader_params.pop("type", None)
    assert reader_type is not None and hasattr(Readers, reader_type),\
        f"Cannot find reader {reader_type}"
    reader = getattr(Readers, reader_type).from_params(dataset_reader_params)
    
    # Load Training Data
    TRAIN_PATH = config.pop("train_data_path")
    logger.info("Loading Training Data from {0}".format(TRAIN_PATH))
    instances_train = reader.read(file_path=TRAIN_PATH)
    logger.info("Length of {0}: {1}".format(
        "Training Data", len(instances_train)))

    # Load Validation Data
    VAL_PATH = config.pop("validation_data_path")
    logger.info("Loading Validation Data from {0}".format(VAL_PATH))
    instances_val = reader.read(VAL_PATH)
    instances_val = instances_val
    logger.info("Length of {0}: {1}".format(
        "Validation Data", len(instances_val)))

    # Load Test Data
    TEST_PATH = config.pop("test_data_path", None)
    instances_test = None
    if TEST_PATH is not None:
        logger.info("Loading Test Data from {0}".format(TEST_PATH))
        instances_test = reader.read(TEST_PATH)
        logger.info("Length of {0}: {1}".format(
            "Testing Data", len(instances_test)))

    # Construct Vocabulary    
    vocab_size = config.pop("max_vocab_size", -1)
    logger.info("Constructing Vocab of size: {0}".format(vocab_size))
    vocab_size = None if vocab_size == -1 else vocab_size
    vocab = Vocabulary.from_instances(
        instances_train,
        max_vocab_size=vocab_size
    )
    vocab_dir = os.path.join(serial_dir, "vocab")
    assert os.path.exists(vocab_dir), "Couldn't find the vocab directory"
    vocab.save_to_files(vocab_dir)
    logger.info("Saving vocab to {0}".format(vocab_dir))
    logger.info("Vocab Construction Done")

    # Construct the data iterators
    logger.info("Constructing Data Iterators")
    data_iterator = DataIterator.from_params(config.pop("iterator"))
    data_iterator.index_with(vocab)

    logger.info("Data Iterators Done")

    # Create the model
    logger.info("Constructing The model")
    model_params = config.pop("model")
    model_type = model_params.pop("type")
    assert model_type is not None and hasattr(Models, model_type),\
        f"Cannot find reader {model_type}"
    model = getattr(Models, model_type).from_params(
        vocab=vocab,
        params=model_params,
    )
    logger.info("Model Construction done")

    # Trainer
    logger.info("Starting the training process")

    serial_dir = os.path.join(serial_dir, "models")

    trainer = Trainer.from_params(
        model=model,
        serialization_dir=serial_dir,
        iterator=data_iterator,
        train_data=instances_train,
        validation_data=instances_val,
        params=config.pop("trainer")
    )
    trainer.train()
    logger.info("Training Done.")
    if instances_test is not None:
        logger.info("Computing final Test Accuracy")
        trainer.test(instances_test)
    logger.info("Done.")


if __name__ == "__main__":
    main()