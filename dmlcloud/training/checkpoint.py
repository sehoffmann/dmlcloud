import json
import logging
import os
from datetime import datetime

import torch.distributed as dist
from wandb.sdk.lib.runid import generate_id

from dmlcloud.config import DefaultConfig


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


def find_old_checkpoint(base_dir):
    slurm_id = get_slurm_id()
    checkpoints = list(base_dir.glob(f'*-{slurm_id} *'))

    if slurm_id and len(checkpoints) == 1:
        # if there exists multiple possible checkpoints, we don't know which one to resume
        # usually only happens for interactive sessions
        model_dir = checkpoints[0]
        job_id = model_dir.name.split(' ')[0]
    else:
        job_id = None
        model_dir = None

    return model_dir, job_id


def sanitize_filename(filename):
    return filename.replace('/', '_')


def generate_job_id():
    slurm_id = get_slurm_id()
    job_id = slurm_id if slurm_id else generate_id()
    date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    job_id = f'{date_str}-{job_id}'
    return job_id


def create_project_dir(base_dir, config):
    job_id = dist.broadcast_object([generate_job_id()])[0]
    dir_name = job_id
    if config.name:
        dir_name += ' ' + sanitize_filename(config.name)

    model_dir = dist.broadcast_object([base_dir / dir_name])[0]

    if dist.rank() == 0:
        os.makedirs(model_dir)
        save_config(get_config_path(model_dir), config)

    return model_dir, job_id


def resume_project_dir(config):
    config.model_dir, config.job_id = find_old_checkpoint(config.checkpoint_dir)
    is_resumed = config.model_dir is not None
    if is_resumed:
        parsed_dct = load_config_dct(get_config_path(config.model_dir))
        consistency_check(parsed_dct, config)
        logging.info(f'Resuming run from {config.model_dir}')
    else:
        config.model_dir, config.job_id = create_project_dir(config.checkpoint_dir, config)
        logging.info(f'Created run directory {config.model_dir}')

    return is_resumed


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
