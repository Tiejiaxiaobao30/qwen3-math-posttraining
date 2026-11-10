import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='/root/autodl-tmp/models/Qwen3-1.7B-Base')
    a = p.parse_args()
    if Path(a.output).exists() and any(Path(a.output).iterdir()):
        raise FileExistsError('Use an empty model directory to avoid mixing revisions')
    from modelscope import snapshot_download
    path = snapshot_download('Qwen/Qwen3-1.7B-Base',
                             revision='43ab5b53bd7f37e7e8b2214adcc9178ee11fb0b3',
                             local_dir=a.output)
    print(path)


if __name__ == '__main__':
    main()
