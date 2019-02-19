# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import argparse
import sys
import os
import logging
import tensorflow as tf

import src.SentenceMatchTrainer as SentenceMatchTrainer
import src.namespace_utils as namespace_utils
from src.SentenceMatchModelGraph import SentenceMatchModelGraph
from src.SentenceMatchDataStream import SentenceMatchDataStream
from src.utils import setup_logger
from src.vocab_utils import Vocab


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, required=True, help='Prefix to the models.')
    parser.add_argument('--in_path', type=str, required=True, help='the path to the test file.')
    parser.add_argument('--out_path', type=str, required=True, help='The path to the output file.')
    parser.add_argument('--word_vec_path', type=str, help='word embedding file for the input file.')

    args, unparsed = parser.parse_known_args()
    setup_logger()
    logger = logging.getLogger(__name__)
    
    # load the configuration file
    log_dir = args.base_dir
    logger.info(f'Loading from base_directory {args.base_dir}')
    config_file = os.path.join(args.base_dir, "config.json")
    options = namespace_utils.load_namespace(config_file)

    if args.word_vec_path is None:
        args.word_vec_path = options.word_vec_path

    # load vocabs
    logger.info('Loading vocabs.')
    word_vocab = Vocab(args.word_vec_path, fileformat='txt3')
    label_vocab_path = os.path.join(
            args.base_dir, "vocab", f"SentenceMatch.{options.suffix}.label_vocab"
        )
    label_vocab = Vocab(label_vocab_path, fileformat='txt2')
    logger.info('word_vocab: {}'.format(word_vocab.word_vecs.shape))
    logger.info('label_vocab: {}'.format(label_vocab.word_vecs.shape))
    num_classes = label_vocab.size()

    if options.with_char:
        char_vocab_path = os.path.join(
            args.base_dir, "vocab", f"SentenceMatch.{options.suffix}.char_vocab"
        )
        char_vocab = Vocab(char_vocab_path, fileformat='txt2')
        logger.info('char_vocab: {}'.format(char_vocab.word_vecs.shape))
    
    logger.info('Build SentenceMatchDataStream ... ')
    testDataStream = SentenceMatchDataStream(args.in_path, word_vocab=word_vocab, char_vocab=char_vocab,
                                            label_vocab=label_vocab,
                                            isShuffle=False, isLoop=True, isSort=True, options=options)
    logger.info('Number of instances in devDataStream: {}'.format(testDataStream.get_num_instance()))
    logger.info('Number of batches in devDataStream: {}'.format(testDataStream.get_num_batch()))
    sys.stdout.flush()

    best_path = os.path.join(
        log_dir, "models", f'SentenceMatch.{options.suffix}.best.model'
    )
    init_scale = 0.01
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        global_step = tf.train.get_or_create_global_step()
        with tf.variable_scope("Model", reuse=False, initializer=initializer):
            valid_graph = SentenceMatchModelGraph(
                num_classes, word_vocab=word_vocab, char_vocab=char_vocab,
                is_training=False, options=options
            )

        initializer = tf.global_variables_initializer()
        vars_ = {}
        for var in tf.global_variables():
            if "word_embedding" in var.name:
                continue
            # if not var.name.startswith("Model"):
            #     continue
            vars_[var.name.split(":")[0]] = var
        saver = tf.train.Saver(vars_)

        sess = tf.Session()
        sess.run(initializer)
        logger.info(f"{best_path}")
        saver.restore(sess, best_path)
        logger.info("DONE!")
        acc = SentenceMatchTrainer.evaluation(sess, valid_graph, testDataStream, outpath=args.out_path,
                                              label_vocab=label_vocab)
        logger.info("Accuracy for test set is %.2f" % acc)


