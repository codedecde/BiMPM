from __future__ import absolute_import
import os
import io
import json
import sys
import logging
import argparse
from subprocess import Popen, PIPE

from src.namespace_utils import Bunch, load_namespace, save_namespace


def bool_flag(s):
    """
        Parse boolean arguments from the command line.

        ..note::
        Usage in argparse:
            parser.add_argument(
                "--cuda", type=bool_flag, default=True, help="Run on GPU")

    """
    if s.lower() in ['off', 'false', '0']:
        return False
    if s.lower() in ['on', 'true', '1']:
        return True
    raise argparse.ArgumentTypeError(
        "invalid value for a boolean flag (0 or 1)")


def setup_logger(logfile: str = "", loglevel: str = "INFO"):
    numeric_level = getattr(logging, loglevel, None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % loglevel)
    logger = logging.getLogger()
    logging.basicConfig(
        format='%(asctime)s: %(levelname)s: %(message)s',
        level=numeric_level, stream=sys.stdout)
    fmt = logging.Formatter('%(asctime)s: %(levelname)s: %(message)s')
    if logfile != "":
        logfile_handle = logging.FileHandler(logfile, 'w')
        logfile_handle.setFormatter(fmt)
        logger.addHandler(logfile_handle)
    return logger


def setup_output_dir(args, loglevel):
    """Setup the Experiment Folder
    Note that the output_dir stores each run as run-1, ....
    Makes the next run directory. This also sets up the logger
    A run directory has the following structure

    * run-1
        * models
                 * modelname*.tar.gz
        * vocab
               * namespace_1.txt
               * namespace_2.txt ...
        * config.json
        * githash.log of current run
        * gitdiff.log of current run
        * logfile.log (the log of the current run)

    This also changes the config, to add the save directory

    Arguments:
        config (``allennlp.common.Params``): The experiment parameters
        loglevel (str): The logger mode [INFO/DEBUG/ERROR]
    Returns
        str, allennlp.common.Params: The filename, and the modified config
    """
    if args.config_path is not None:
        FLAGS = load_namespace(args.config_path)
    else:
        FLAGS = Bunch(vars(args))
    output_dir = FLAGS.base_output_dir
    make_directory(output_dir, recursive=True)
    last_run = -1
    for dirname in os.listdir(output_dir):
        if dirname.startswith('run-'):
            last_run = max(last_run, int(dirname.split('-')[1]))
    new_dirname = os.path.join(output_dir, 'run-%d' % (last_run + 1))
    make_directory(new_dirname)
    best_model_dirname = os.path.join(new_dirname, 'models')
    make_directory(best_model_dirname)
    vocab_dirname = os.path.join(new_dirname, 'vocab')
    make_directory(vocab_dirname)

    config_file = os.path.join(new_dirname, 'config.json')
    save_namespace(FLAGS, config_file)

    # Save the git hash
    process = Popen(
        'git log -1 --format="%H"'.split(), stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    stdout = stdout.decode('ascii').strip('\n').strip('"')
    with open(os.path.join(new_dirname, "githash.log"), "w") as fp:
        fp.write(stdout)

    # Save the git diff
    process = Popen('git diff'.split(), stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    with open(os.path.join(new_dirname, "gitdiff.log"), "w") as fp:
        stdout = stdout.decode('ascii')
        fp.write(stdout)

    # Set up the logger
    logfile = os.path.join(new_dirname, 'logfile.log')
    setup_logger(logfile, loglevel)
    logger = logging.getLogger(__name__)
    logger.info(f"Read config from {args.config_path}")

    # modify experiment path
    setattr(FLAGS, "base_output_dir", new_dirname)
    return FLAGS


def write_config_to_file(filepath, config):
    """Writes the config to a json file, specifed by filepath
    """
    with io.open(filepath, 'w', encoding='utf-8', errors='ignore') as fd:
        json.dump(fp=fd,
                  obj=config.as_dict(quiet=True),
                  ensure_ascii=False, indent=4, sort_keys=True)


def make_directory(dirname, recursive=False):
    """Constructs a directory with name dirname, if
    it doesn't exist. Can also take in a path, and recursively
    apply it.

    .. note::
        The recursive directory structure may cause issues on a windows
        system.
    """
    try:
        os.makedirs(dirname, exist_ok=True)
    except OSError:
        raise
