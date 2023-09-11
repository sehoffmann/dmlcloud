import json
import logging
import os
from datetime import datetime

import horovod.torch as hvd
from wandb.sdk.lib.runid import generate_id

from .config import DefaultConfig


class ExtendedJSONEncoder(json.JSONEncoder):
    """
    JSONEncoder subclass that serializes classes and functions as well (by their name).
    """

    def default(self, o):
        if isinstance(o, type):
            return f'<cls {o.__module__}.{o.__name__}>'
        elif callable(o):
            return f'<fn {o.__module__}.{o.__name__}>'

        try:
            return super().default(o)
        except TypeError:
            return str(o)


class ExtendedJSONDecoder(json.JSONDecoder):
    pass


def get_config_path(model_dir):
    return model_dir / 'config.json'


def get_checkpoint_path(model_dir):
    return model_dir / 'checkpoint.pth'


def get_slurm_id():
    return os.environ.get('SLURM_JOB_ID')


def find_old_checkpoint(base_dir, id_prefix):
    slurm_id = get_slurm_id()
    slurm_dir = next(iter(base_dir.glob(f'*-{id_prefix}{slurm_id}')), None)

    if get_config_path(base_dir).exists():
        model_dir = base_dir
        job_id = base_dir.stem.split('-', 1)[0]
    elif slurm_id and slurm_dir is not None:
        model_dir = slurm_dir
        job_id = id_prefix + slurm_id
    else:
        job_id = None
        model_dir = None

    return model_dir, job_id


def sanitize_filename(filename):
    return filename.replace('/', '_')


def create_project_dir(base_dir, config):
    slurm_id = get_slurm_id()
    job_id = hvd.broadcast_object(slurm_id if slurm_id else generate_id(), name='job_id')
    job_id = config.id_prefix + job_id

    date_str = datetime.now().strftime('%Y%m%d_%H%M%S')

    name = f'{date_str}-{job_id}'
    if config.wb_experiment:
        name += ' ' + sanitize_filename(config.wb_experiment)

    model_dir = hvd.broadcast_object(base_dir / name, name='model_dir')

    if hvd.rank() == 0:
        os.makedirs(model_dir)
        save_config(get_config_path(model_dir), config)

    return model_dir, job_id


def resume_project_dir(base_dir, config):
    model_dir, job_id = find_old_checkpoint(base_dir, config.id_prefix)
    if model_dir is not None:
        parsed_dct = load_config_dct(get_config_path(model_dir))
        consistency_check(parsed_dct, config)
        is_resumed = True
        logging.info(f'Resuming run from {model_dir}')
    else:
        model_dir, job_id = create_project_dir(base_dir, config)
        is_resumed = False
        logging.info(f'Created run directory {model_dir}')
    return model_dir, job_id, is_resumed


def consistency_check(parsed_dct, config):
    parsed_cfg = DefaultConfig(parsed_dct)
    if parsed_cfg.git_hash != config.git_hash:
        msg = 'Git hash of resumed run does not match current git hash.\n'
        msg += f'Current git hash: {config.git_hash}\n'
        msg += f'Git hash of resumed run: {parsed_cfg.git_hash}'
        logging.warning(msg)

    if parsed_cfg.command_line != config.command_line:
        msg = 'Command line of resumed run does not match current command line.\n'
        msg += f'Current command line: {config.command_line}\n'
        msg += f'Command line of resumed run: {parsed_cfg.command_line}'
        logging.warning(msg)


def save_config(path, config):
    with open(path, 'w') as file:
        json.dump(config.as_dictionary(), file, cls=ExtendedJSONEncoder, indent=4)


def load_config_dct(path):
    with open(path) as file:
        dct = json.load(file, cls=ExtendedJSONDecoder)
        return dct
