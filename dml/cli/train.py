import argparse

from ..experiments import ALL_EXPERIMENTS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--interactive', action='store_true', help='If set, dont use wandb and checkpointing')
    parser.add_argument('-n', '--steps', type=int, default=None, help='Number of steps per epoch')
    subparsers = parser.add_subparsers(help='Experiment to run')
    subparsers.required = True

    for exp in ALL_EXPERIMENTS:
        exp.add_parser(subparsers)

    args = parser.parse_args()
    trainer = args.create_trainer(args)
    return args, trainer


def main():
    args, trainer = parse_args()
    trainer.train(max_steps=args.steps, use_checkpointing=not args.interactive, use_wandb=not args.interactive)


if __name__ == '__main__':
    main()
