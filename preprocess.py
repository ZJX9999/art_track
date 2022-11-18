import os
import sys

from src.args_util import command, create_arg_parser, TARGET_MAT2JSON, TARGET_MPII_SINGLE, TARGET_PAIRWISE, TARGET_TF2MS


@command
def preprocess(parser, args, cfg):
    if args.target == TARGET_MPII_SINGLE:
        a = (args.dataset_dir, args.dataset_name, args.save_dir, args.image_dir)
        from src.tool.preprocess.preprocess_single import preprocess_single
        preprocess_single(*a)

        if args.split:
            if args.save_dir is not None:
                _dir = args.save_dir
            elif args.dataset_dir is not None:
                _dir = os.path.join(args.dataset_dir, 'cropped')
            else:
                _dir = './cropped'
            from src.tool.preprocess.split import split
            split(os.path.join(_dir, 'dataset.json'), _dir)
    elif args.target == TARGET_TF2MS:
        a = (args.checkpoint, args.output, args.map)
        from src.tool.preprocess.tf2ms import tf2ms
        tf2ms(*a)
    elif args.target == TARGET_MAT2JSON:
        from src.tool.preprocess.mat2json import mat2json
        mat2json(args.index_mat, args.name, args.dataset_json, args.output_dir, args.index_offset, args.stdout)
    elif args.target == TARGET_PAIRWISE:
        from src.tool.preprocess.pairwise_stats import pairwise_stats
        pairwise_stats(cfg)
    else:
        parser.print_help()


def main():
    parser = create_arg_parser()['preprocess']
    args = parser.parse_args(sys.argv[1:])
    preprocess(parser, args)


if __name__ == '__main__':
    main()
