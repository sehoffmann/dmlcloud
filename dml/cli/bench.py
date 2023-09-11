import argparse

from ..experiments import ALL_EXPERIMENTS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--steps', type=int, default=200, help='Number of steps to benchmark')
    subparsers = parser.add_subparsers(help='Experiment to run')
    subparsers.required = True

    for exp in ALL_EXPERIMENTS:
        exp.add_parser(subparsers)

    args = parser.parse_args()
    trainer = args.create_trainer(args)
    return args, trainer


def main():
    args, trainer = parse_args()
    print(f'Benchmarking {args.steps} steps')
    trainer.cfg.epochs = 1
    trainer.train(args.steps, use_checkpointing=False, use_wandb=False, print_diagnostics=False)
