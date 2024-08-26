from pathlib import Path


def make_outputdir_and_log(args):
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        if not Path(args.output_dir + '/train_log.txt').exists():
            open(args.output_dir + '/train_log.txt', mode='w')
        if not Path(args.output_dir + '/val_log.txt').exists():
            open(args.output_dir + '/val_log.txt', mode='w')
        Path(args.output_dir + '/val_visualize').mkdir(parents=True, exist_ok=True)
        if not Path(args.output_dir + '/args.txt').exists():
            f_args = open(args.output_dir + '/args.txt', mode='w')
            f_args.write(args.__str__())
