import argparse
import os
import shutil


def collect_metrics(base_dir: str) -> None:
    for dirname, dir_dirs, dir_files in os.walk(base_dir):
        if not dirname.endswith('tensorboard'):
            continue
        tb_files = []
        for tb_dir in dir_dirs:
            if len(tb_files) == 0:
                tb_files.append(os.path.join(dirname, tb_dir, 'hparams.yaml'))
            tb_files.extend([os.path.join(dirname, tb_dir, file)
                             for file in os.listdir(os.path.join(dirname, tb_dir))
                             if not file.endswith('hparams.yaml')])
        for tb_file in tb_files:
            shutil.copy2(tb_file, os.path.join(dirname, os.path.basename(tb_file)))
        for tb_dir in dir_dirs:
            shutil.rmtree(os.path.join(dirname, tb_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Collect Tensorboard Logs (for trainigs made with this framework ONLY!)',
    )
    parser.add_argument(
        'base_directory',
        type=str,
        help='Base directory to search for `tensorboard` directories beneath.',
    )
    args = parser.parse_args()
    collect_metrics(args.base_directory)
