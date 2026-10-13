"""Entry point for OpenDepth robust, single-step training."""
from finetuning_d_robust import Step1XEditRobustDiscriminativeTrainer, setup_parser
from library import train_util


def main():
    parser = setup_parser()
    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)
    args.discriminative_mode = True
    Step1XEditRobustDiscriminativeTrainer().train(args)


if __name__ == '__main__':
    main()
